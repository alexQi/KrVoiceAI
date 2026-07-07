"""数字人 API 服务（云端 GPU 部署）

支持两种推理后端：
- LatentSync 1.5：字节跳动开源，潜在扩散模型，口型质量高（推荐，provider=latentsync）
- MuseTalk：腾讯音乐开源，实时唇同步（备选，provider=musetalk）

在云 GPU 上启动此服务，提供数字人口播生成 API。
本地 EnlyAI 通过 GPURunner 调用此服务。

启动方式（用 LatentSync 环境的 python，确保子进程能跑通官方推理脚本）：
    LATENTSYNC_DIR=/path/to/LatentSync \
    AVATAR_BACKEND=latentsync CUDA_VISIBLE_DEVICES=0 \
    python -m krvoiceai.api.avatar_server --port 8010 --backend latentsync

依赖（云端安装，参考 scripts/setup_cloud_gpu.sh 一键安装）：
    pip install fastapi uvicorn
    # LatentSync（推荐，本服务用子进程调其 scripts/inference.py，对版本鲁棒）：
    git clone https://github.com/bytedance/LatentSync.git
    cd LatentSync && pip install -e . && bash setup_env.sh   # setup_env.sh 下载 UNet/whisper 权重
    # 关键环境变量：LATENTSYNC_DIR（仓库根）/ LATENTSYNC_CKPT / LATENTSYNC_CONFIG
    #             LATENTSYNC_CONFIG_512（可选高清配置）/ LATENTSYNC_GUIDANCE（默认 1.5）
    # 或 MuseTalk（备选）：
    pip install musetalk opencv-python
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="EnlyAI Avatar Server", version="2.0")

# 推理后端实例（延迟加载）
_avatar_backend: Optional[Any] = None
_avatars_dir = Path(os.environ.get("AVATARS_DIR", "./config/avatars"))
_backend_name = os.environ.get("AVATAR_BACKEND", "latentsync")  # latentsync / musetalk


class GenerateRequest(BaseModel):
    """数字人生成请求"""
    audio_base64: str
    avatar_id: str = "default"
    output_fps: int = 25
    output_resolution: list[int] = [1080, 1920]
    # LatentSync 专用参数（可选，覆盖默认）
    inference_steps: Optional[int] = None
    resolution: Optional[int] = None
    config_name: Optional[str] = None  # high_quality / fast


class RegisterRequest(BaseModel):
    """数字人形象注册请求"""
    avatar_id: str
    reference_video_base64: str


def _get_avatar_backend():
    """延迟加载推理后端

    返回一个统一接口对象，暴露 generate(audio_path, avatar_id, **kwargs) -> video_path。
    未安装对应依赖时返回 None（使用 ffmpeg 占位降级）。
    """
    global _avatar_backend
    if _avatar_backend is not None:
        return _avatar_backend

    if _backend_name == "latentsync":
        _avatar_backend = _load_latentsync()
    elif _backend_name == "musetalk":
        _avatar_backend = _load_musetalk()
    else:
        print(f"[warn] 未知后端 {_backend_name}，使用占位实现")

    return _avatar_backend


def _load_latentsync():
    """加载 LatentSync 1.5 后端（子进程方式调用官方 scripts/inference.py）

    LatentSync 是字节跳动开源的潜在扩散唇同步模型，口型质量超过 MuseTalk。
    它不是 pip 包，而是一个 Git 仓库（config yaml + checkpoint + scripts/inference.py）。
    这里用「子进程调官方推理脚本」而非 import 内部类，对版本变动最鲁棒。

    需要的环境变量（部署时设置，见 docs/GPU_UPGRADE.md §2）：
      LATENTSYNC_DIR     LatentSync 仓库根目录（默认 ./LatentSync）
      LATENTSYNC_CKPT    UNet 权重（默认 <DIR>/checkpoints/latentsync_unet.pt）
      LATENTSYNC_CONFIG  UNet 配置 yaml（默认 <DIR>/configs/unet/stage2.yaml）
      LATENTSYNC_CONFIG_512  高清 512 配置（可选，config_name=high_quality 时优先用）
      LATENTSYNC_GUIDANCE    guidance_scale（默认 1.5）
    参考：https://github.com/bytedance/LatentSync
    """
    repo_dir = Path(os.environ.get("LATENTSYNC_DIR", "./LatentSync")).resolve()
    ckpt = Path(os.environ.get(
        "LATENTSYNC_CKPT", str(repo_dir / "checkpoints" / "latentsync_unet.pt")
    ))
    config = Path(os.environ.get(
        "LATENTSYNC_CONFIG", str(repo_dir / "configs" / "unet" / "stage2.yaml")
    ))
    inference_module = repo_dir / "scripts" / "inference.py"

    missing = [str(p) for p in (repo_dir, inference_module, ckpt, config) if not p.exists()]
    if missing:
        print(f"[warn] LatentSync 未就绪，缺少：{missing}，使用 ffmpeg 占位降级")
        print("[info] 安装：git clone https://github.com/bytedance/LatentSync && "
              "cd LatentSync && pip install -e . && bash setup_env.sh（下载权重）")
        print("[info] 设置 LATENTSYNC_DIR 指向仓库根目录")
        return None

    print(f"[info] LatentSync 后端就绪 repo={repo_dir} ckpt={ckpt.name} config={config.name}")
    return _LatentSyncWrapper(repo_dir, ckpt, config)


class _LatentSyncWrapper:
    """LatentSync 统一接口封装（子进程调用官方 scripts/inference.py）"""

    def __init__(self, repo_dir: Path, ckpt: Path, config: Path):
        self.repo_dir = repo_dir
        self.ckpt = ckpt
        self.config = config
        self.config_512 = os.environ.get("LATENTSYNC_CONFIG_512", "")
        self.guidance = os.environ.get("LATENTSYNC_GUIDANCE", "1.5")

    def generate(
        self, audio_path: str, avatar_id: str,
        output_path: str, inference_steps: int = 25,
        resolution: int = 512, config_name: str | None = None, **kwargs,
    ) -> str:
        """调用 LatentSync 官方推理脚本

        Args:
            audio_path: 输入音频路径（wav）
            avatar_id: 形象 ID（对应 _avatars_dir/<id>/reference.mp4）
            output_path: 输出视频路径
            inference_steps: 扩散步数（20-25 平衡，50 最高，10 最快）
            resolution: 处理分辨率（512 高清 / 256 更快）；仅用于选配置文件
            config_name: high_quality / fast（high_quality 优先用 512 配置）

        Returns:
            输出视频路径
        """
        ref_video = self._get_reference_video(avatar_id)

        # 高清 512 配置：显式指定 config_name=high_quality 或 resolution>=512 时优先
        unet_config = self.config
        want_hq = (config_name == "high_quality") or (resolution and resolution >= 512)
        if want_hq and self.config_512 and Path(self.config_512).exists():
            unet_config = Path(self.config_512)

        cmd = [
            sys.executable, "-m", "scripts.inference",
            "--unet_config_path", str(unet_config),
            "--inference_ckpt_path", str(self.ckpt),
            "--inference_steps", str(int(inference_steps)),
            "--guidance_scale", str(self.guidance),
            "--video_path", str(ref_video),
            "--audio_path", str(audio_path),
            "--video_out_path", str(output_path),
        ]
        seed = kwargs.get("seed")
        if seed is not None and int(seed) >= 0:
            cmd += ["--seed", str(int(seed))]

        print(f"[info] LatentSync 推理: steps={inference_steps} "
              f"config={unet_config.name} ref={Path(ref_video).name}")
        r = subprocess.run(
            cmd, cwd=str(self.repo_dir),
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not Path(output_path).exists():
            raise RuntimeError(
                f"LatentSync 推理失败 rc={r.returncode}: {r.stderr[-800:]}"
            )
        return output_path

    @staticmethod
    def _get_reference_video(avatar_id: str) -> str:
        avatar_dir = _avatars_dir / avatar_id
        for name in ("reference.mp4", "ref.mp4", "avatar.mp4"):
            p = avatar_dir / name
            if p.exists():
                return str(p)
        raise FileNotFoundError(f"形象 {avatar_id} 未注册参考视频")


def _load_musetalk():
    """加载 MuseTalk 后端（备选）"""
    try:
        from musetalk.api import MuseTalkAPI
        model = MuseTalkAPI(avatar_path=str(_avatars_dir))
        print("[info] MuseTalk 后端加载成功")
        return _MuseTalkWrapper(model)
    except ImportError:
        print("[warn] MuseTalk 未安装，使用 ffmpeg 占位降级")
        return None


class _MuseTalkWrapper:
    """MuseTalk 统一接口封装"""

    def __init__(self, model):
        self.model = model

    def generate(
        self, audio_path: str, avatar_id: str,
        output_path: str, **kwargs,
    ) -> str:
        # MuseTalk 真实调用（接口根据版本调整）
        result = self.model.generate(
            audio_path=audio_path,
            avatar_id=avatar_id,
            fps=kwargs.get("output_fps", 25),
        )
        # MuseTalk 返回的是临时文件路径，复制到 output_path
        import shutil
        shutil.copy2(result, output_path)
        return output_path


@app.get("/health")
def health():
    """健康检查（含后端就绪状态）"""
    backend = _get_avatar_backend()
    return {
        "status": "ok",
        "service": "avatar",
        "backend": _backend_name,
        "backend_ready": backend is not None,
    }


@app.post("/api/avatar/generate")
def generate(req: GenerateRequest):
    """生成数字人口播视频

    Args:
        req: 包含音频 base64 和 avatar_id

    Returns:
        video_base64: 生成的视频（base64 编码）
        duration: 视频时长（秒）
        backend: 实际使用的后端（latentsync/musetalk/placeholder）
    """
    try:
        # 先查找形象参考视频（不依赖模型加载）
        avatar_dir = _avatars_dir / req.avatar_id
        ref_video = None
        for name in ("reference.mp4", "ref.mp4", "avatar.mp4"):
            p = avatar_dir / name
            if p.exists():
                ref_video = p
                break

        if not ref_video:
            raise HTTPException(
                status_code=404,
                detail=f"形象 {req.avatar_id} 未注册",
            )

        backend = _get_avatar_backend()

        # 解码音频到临时文件
        audio_bytes = base64.b64decode(req.audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            audio_path = f.name

        try:
            output_path = tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False
            ).name

            backend_used = "placeholder"
            if backend is not None:
                # 真实推理调用
                try:
                    backend.generate(
                        audio_path=audio_path,
                        avatar_id=req.avatar_id,
                        output_path=output_path,
                        inference_steps=req.inference_steps or 25,
                        resolution=req.resolution or 512,
                        config_name=req.config_name,
                        output_fps=req.output_fps,
                    )
                    backend_used = _backend_name
                except Exception as e:
                    print(f"[error] {_backend_name} 推理失败，降级占位：{e}")
                    _placeholder_generate(ref_video, audio_path, output_path)
            else:
                # 占位实现：ffmpeg 把参考视频 + 音频合成视频
                _placeholder_generate(ref_video, audio_path, output_path)

            video_bytes = Path(output_path).read_bytes()
            video_b64 = base64.b64encode(video_bytes).decode()

            # 探测时长
            duration = _probe_duration(output_path)

            return {
                "video_base64": video_b64,
                "duration": duration,
                "avatar_id": req.avatar_id,
                "backend": backend_used,
            }
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _placeholder_generate(ref_video: Path, audio_path: str, output_path: str):
    """占位实现：用 ffmpeg 把参考视频 + 音频合成视频（无唇同步，仅供流程跑通）"""
    import subprocess
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(ref_video),
            "-i", audio_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ],
        capture_output=True, check=True,
    )


@app.post("/api/avatar/register")
def register(req: RegisterRequest):
    """注册数字人形象

    将参考视频保存到 avatars_dir/<avatar_id>/reference.mp4
    """
    try:
        avatar_dir = _avatars_dir / req.avatar_id
        avatar_dir.mkdir(parents=True, exist_ok=True)
        video_bytes = base64.b64decode(req.reference_video_base64)
        ref_path = avatar_dir / "reference.mp4"
        ref_path.write_bytes(video_bytes)

        # 抽取首帧作为预览图
        try:
            import subprocess
            preview = avatar_dir / "reference.jpg"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(ref_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(preview),
                ],
                capture_output=True, check=True,
            )
        except Exception:
            pass

        # 保存元数据
        import json
        (avatar_dir / "meta.json").write_text(
            json.dumps({
                "avatar_id": req.avatar_id,
                "source": "cloud_register",
                "mode": _backend_name,
                "registered_at": time.time(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {"success": True, "avatar_id": req.avatar_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _probe_duration(video_path: str) -> float:
    """探测视频时长"""
    try:
        import subprocess
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="EnlyAI Avatar Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--backend", default="latentsync",
        choices=["latentsync", "musetalk"],
        help="推理后端（latentsync 推荐，musetalk 备选）",
    )
    args = parser.parse_args()

    global _backend_name
    _backend_name = args.backend
    os.environ["AVATAR_BACKEND"] = args.backend

    # 预热后端（启动时即加载，避免首个请求慢）
    backend = _get_avatar_backend()
    status = "就绪" if backend is not None else "占位降级（未装推理依赖）"
    print(f"数字人服务启动: http://{args.host}:{args.port}")
    print(f"后端: {args.backend} [{status}]")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
