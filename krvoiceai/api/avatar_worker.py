"""数字人常驻推理 Worker（LatentSync，单卡一进程，模型只加载一次）

M1：多卡并行生成单条视频的地基。相比 avatar_server.py「每请求子进程冷加载」，
本 worker 启动时把 LatentSync 加载一次常驻显存，提供**段级**推理接口，供上层
调度器（M2）把一条视频的多段并行分发到多张卡。

启动（每卡一个进程，用 CUDA_VISIBLE_DEVICES 钉卡）：
    LATENTSYNC_DIR=/path/to/LatentSync CUDA_VISIBLE_DEVICES=0 \
    python -m krvoiceai.api.avatar_worker --port 8010

推理模式（LATENTSYNC_MODE，默认 resident）：
    resident   —— 进程内加载模型一次，段级推理复用（多卡分片必须用此模式）
    subprocess —— 回退：每请求调官方 scripts/inference.py（能跑但每次冷加载，慢）
    resident 加载失败会**自动回退 subprocess**并打印告警；两者都不可用时回退 ffmpeg 占位。

关键环境变量：
    LATENTSYNC_DIR / LATENTSYNC_CKPT / LATENTSYNC_CONFIG / LATENTSYNC_CONFIG_512
    LATENTSYNC_WHISPER_CKPT（resident 模式的 whisper 权重，默认 <DIR>/checkpoints/whisper/tiny.pt）
    LATENTSYNC_GUIDANCE（默认 1.5） / AVATARS_DIR（形象库，默认 ./config/avatars）

> resident 进程内加载对齐 LatentSync 官方 scripts/inference.py。不同版本构造/调用
> 可能有出入——若启动日志显示回退到 subprocess，把你的 scripts/inference.py 内容发来，
> 即可把 _load_resident/_infer_resident 精确对齐你的版本。
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="EnlyAI Avatar Worker", version="1.0")

_avatars_dir = Path(os.environ.get("AVATARS_DIR", "./config/avatars"))
_engine: Optional["LatentSyncEngine"] = None

# GPU 串行化：单卡一次只跑一段，避免并发打爆显存
_gpu_lock = threading.Semaphore(1)
_stat_lock = threading.Lock()
_inflight = 0   # 正在推理
_queued = 0     # 已进入端点、等待 GPU 锁


# ────────────────────────── 请求模型 ──────────────────────────

class GenerateSegmentRequest(BaseModel):
    """段级推理请求（多卡分片用）"""
    audio_base64: str                 # 该段音频（wav）
    avatar_id: str = "default"
    ref_start: Optional[float] = None  # 参考视频切片起点（秒）；None=整段参考
    ref_end: Optional[float] = None    # 参考视频切片终点（秒）
    seg_index: int = 0
    inference_steps: int = 25
    resolution: int = 512
    config_name: Optional[str] = None  # high_quality / fast
    seed: int = 1247                   # 固定 seed，保证各段风格一致、可复现


class GenerateRequest(BaseModel):
    """整条推理请求（兼容 avatar_server 老契约，非分片路径）"""
    audio_base64: str
    avatar_id: str = "default"
    output_fps: int = 25
    output_resolution: list[int] = [1080, 1920]
    inference_steps: Optional[int] = None
    resolution: Optional[int] = None
    config_name: Optional[str] = None
    seed: int = 1247


class RegisterRequest(BaseModel):
    avatar_id: str
    reference_video_base64: str


# ────────────────────────── 推理引擎 ──────────────────────────

class LatentSyncEngine:
    """LatentSync 推理引擎：resident（进程内常驻）/ subprocess（冷加载回退）/ placeholder。

    统一接口：infer(ref_video, audio_path, out_path, steps, resolution, seed, config_name)
    """

    def __init__(self):
        self.repo_dir = Path(os.environ.get("LATENTSYNC_DIR", "./LatentSync")).resolve()
        self.ckpt = Path(os.environ.get(
            "LATENTSYNC_CKPT", str(self.repo_dir / "checkpoints" / "latentsync_unet.pt")))
        self.config = Path(os.environ.get(
            "LATENTSYNC_CONFIG", str(self.repo_dir / "configs" / "unet" / "stage2.yaml")))
        self.config_512 = os.environ.get("LATENTSYNC_CONFIG_512", "")
        self.whisper_ckpt = Path(os.environ.get(
            "LATENTSYNC_WHISPER_CKPT", str(self.repo_dir / "checkpoints" / "whisper" / "tiny.pt")))
        self.guidance = float(os.environ.get("LATENTSYNC_GUIDANCE", "1.5"))
        self.mode = os.environ.get("LATENTSYNC_MODE", "resident").lower()
        self._pipe = None        # resident pipeline 对象
        self._num_frames = 16    # 由 config 覆盖

    # ---- 加载 ----
    def load(self) -> None:
        repo_ok = self.repo_dir.exists() and self.ckpt.exists() and self.config.exists()
        if not repo_ok:
            print(f"[warn] LatentSync 未就绪（repo/ckpt/config 缺失于 {self.repo_dir}），"
                  f"worker 以 placeholder 模式启动（仅供 M2/M3 编排联调）")
            self.mode = "placeholder"
            return

        if self.mode == "resident":
            try:
                self._load_resident()
                print(f"[info] LatentSync resident 加载成功（模型常驻显存，num_frames={self._num_frames}）")
                return
            except Exception as e:
                print(f"[warn] resident 进程内加载失败：{e}")
                print("[warn] 自动回退 subprocess 模式（每请求冷加载，慢）。"
                      "把你的 LatentSync scripts/inference.py 发来即可精确对齐 resident 加载。")
                self.mode = "subprocess"

        if self.mode == "subprocess":
            inf = self.repo_dir / "scripts" / "inference.py"
            if not inf.exists():
                print(f"[warn] {inf} 不存在，回退 placeholder")
                self.mode = "placeholder"
            else:
                print("[info] LatentSync subprocess 模式就绪（scripts/inference.py，逐请求冷加载）")

    def _load_resident(self) -> None:
        """进程内加载 LatentSync（对齐官方 scripts/inference.py 的构建流程）。

        不同 LatentSync 版本构造签名可能不同；本方法失败会被 load() 捕获并回退 subprocess。
        """
        import torch
        from omegaconf import OmegaConf
        from diffusers import AutoencoderKL, DDIMScheduler

        # 把仓库根加入 import 路径
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        # LatentSync 用相对路径读资产（如 latentsync/utils/mask.png），必须把 CWD 切到仓库根。
        # AVATARS_DIR/临时文件均为绝对路径，chdir 不影响它们。
        os.chdir(str(self.repo_dir))
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature

        cfg = OmegaConf.load(str(self.config))
        self._num_frames = int(cfg.data.num_frames)
        self._resolution = int(cfg.data.resolution)
        self._mask_path = cfg.data.get("mask_image_path", "latentsync/utils/mask.png")
        self._dtype = torch.float16
        device = "cuda"

        scheduler = DDIMScheduler.from_pretrained(str(self.repo_dir / "configs"))
        audio_encoder = Audio2Feature(
            model_path=str(self.whisper_ckpt), device=device,
            num_frames=self._num_frames,
            audio_feat_length=cfg.data.get("audio_feat_length", [2, 2]))
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self._dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0  # 官方 scripts/inference.py 必设；缺则 prepare_mask_latents 报 Tensor-None
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(cfg.model), str(self.ckpt), device="cpu")
        unet = unet.to(dtype=self._dtype)
        self._pipe = LipsyncPipeline(
            vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler,
        ).to(device)

    @property
    def ready(self) -> bool:
        return self.mode in ("resident", "subprocess") or self.mode == "placeholder"

    # ---- 推理（GPU 段推理，已在 GPU 锁内调用）----
    def infer(self, ref_video: str, audio_path: str, out_path: str,
              inference_steps: int, resolution: int, seed: int,
              config_name: Optional[str]) -> None:
        if self.mode == "resident":
            self._infer_resident(ref_video, audio_path, out_path, inference_steps, resolution, seed)
        elif self.mode == "subprocess":
            self._infer_subprocess(ref_video, audio_path, out_path, inference_steps, resolution, seed, config_name)
        else:
            _placeholder_generate(ref_video, audio_path, out_path)

    def _infer_resident(self, ref_video, audio_path, out_path, steps, resolution, seed) -> None:
        # 严格对齐官方 scripts/inference.py 的 pipeline(...) 调用（宽高用 config.data.resolution，
        # 传 mask_image_path/temp_dir，不传 seed——LipsyncPipeline 用 generator 而非 seed）
        temp_dir = tempfile.mkdtemp(prefix="latentsync_")
        self._pipe(
            video_path=str(ref_video),
            audio_path=str(audio_path),
            video_out_path=str(out_path),
            num_frames=self._num_frames,
            num_inference_steps=int(steps),
            guidance_scale=self.guidance,
            weight_dtype=self._dtype,
            width=self._resolution,
            height=self._resolution,
            mask_image_path=self._mask_path,
            temp_dir=temp_dir,
        )
        if not Path(out_path).exists():
            raise RuntimeError("LatentSync resident 推理未产出视频")

    def _infer_subprocess(self, ref_video, audio_path, out_path, steps, resolution, seed, config_name) -> None:
        unet_config = self.config
        want_hq = (config_name == "high_quality") or (resolution and resolution >= 512)
        if want_hq and self.config_512 and Path(self.config_512).exists():
            unet_config = Path(self.config_512)
        cmd = [
            sys.executable, "-m", "scripts.inference",
            "--unet_config_path", str(unet_config),
            "--inference_ckpt_path", str(self.ckpt),
            "--inference_steps", str(int(steps)),
            "--guidance_scale", str(self.guidance),
            "--video_path", str(ref_video),
            "--audio_path", str(audio_path),
            "--video_out_path", str(out_path),
            "--seed", str(int(seed)),
        ]
        r = subprocess.run(cmd, cwd=str(self.repo_dir), capture_output=True, text=True)
        if r.returncode != 0 or not Path(out_path).exists():
            raise RuntimeError(f"LatentSync subprocess 推理失败 rc={r.returncode}: {r.stderr[-800:]}")


# ────────────────────────── 辅助 ──────────────────────────

def _get_reference_video(avatar_id: str) -> Path:
    avatar_dir = _avatars_dir / avatar_id
    for name in ("reference.mp4", "ref.mp4", "avatar.mp4"):
        p = avatar_dir / name
        if p.exists():
            return p
    raise HTTPException(status_code=404, detail=f"形象 {avatar_id} 未注册参考视频")


def _slice_reference(ref: Path, start: Optional[float], end: Optional[float]) -> tuple[Path, bool]:
    """按时间窗口切参考视频；返回 (路径, 是否为临时切片)。start/end 为 None 时返回整段。

    重编码保证切点帧准确、GOP 干净（每段短，一次性开销可接受）。
    """
    if start is None or end is None or end <= start:
        return ref, False
    dur = end - start
    out = Path(tempfile.mktemp(suffix=".mp4"))
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(ref), "-t", f"{dur:.3f}",
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"参考视频切片失败 [{start:.2f},{end:.2f}]: {r.stderr[-400:]}")
    return out, True


def _placeholder_generate(ref_video, audio_path: str, output_path: str) -> None:
    """占位：ffmpeg 把参考视频+音频合成（无唇同步，仅供编排联调）"""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(ref_video), "-i", audio_path,
         "-c:v", "libx264", "-c:a", "aac", "-shortest", output_path],
        capture_output=True, check=True,
    )


def _probe_duration(video_path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _run_inference(ref_video: Path, audio_path: str, out_path: str,
                   steps: int, resolution: int, seed: int, config_name: Optional[str]) -> None:
    """在 GPU 锁内串行执行一次推理，维护 inflight/queued 计数。"""
    global _inflight, _queued
    with _stat_lock:
        _queued += 1
    _gpu_lock.acquire()
    with _stat_lock:
        _queued -= 1
        _inflight += 1
    try:
        _engine.infer(str(ref_video), audio_path, out_path, steps, resolution, seed, config_name)
    finally:
        with _stat_lock:
            _inflight -= 1
        _gpu_lock.release()


# ────────────────────────── 路由 ──────────────────────────

@app.get("/health")
def health():
    with _stat_lock:
        inflight, queued = _inflight, _queued
    return {
        "status": "ok", "service": "avatar_worker", "backend": "latentsync",
        "mode": _engine.mode if _engine else "unloaded",
        "backend_ready": bool(_engine and _engine.ready),
        "inflight": inflight, "queued": queued,
    }


@app.post("/api/avatar/generate_segment")
def generate_segment(req: GenerateSegmentRequest):
    """段级唇形推理：切参考窗口 → 对(切片,段音频)推理 → 返回段视频"""
    ref = _get_reference_video(req.avatar_id)
    audio_bytes = base64.b64decode(req.audio_base64)
    audio_path = tempfile.mktemp(suffix=".wav")
    Path(audio_path).write_bytes(audio_bytes)
    out_path = tempfile.mktemp(suffix=".mp4")
    ref_slice, is_tmp = ref, False
    try:
        ref_slice, is_tmp = _slice_reference(ref, req.ref_start, req.ref_end)
        _run_inference(ref_slice, audio_path, out_path,
                       req.inference_steps, req.resolution, req.seed, req.config_name)
        video_b64 = base64.b64encode(Path(out_path).read_bytes()).decode()
        return {
            "video_base64": video_b64, "seg_index": req.seg_index,
            "duration": _probe_duration(out_path), "mode": _engine.mode,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"seg {req.seg_index} 失败: {e}")
    finally:
        for p in (audio_path, out_path):
            if os.path.exists(p):
                os.unlink(p)
        if is_tmp and os.path.exists(ref_slice):
            os.unlink(ref_slice)


@app.post("/api/avatar/generate")
def generate(req: GenerateRequest):
    """整条推理（兼容老契约，用常驻模型；不分片）"""
    ref = _get_reference_video(req.avatar_id)
    audio_path = tempfile.mktemp(suffix=".wav")
    Path(audio_path).write_bytes(base64.b64decode(req.audio_base64))
    out_path = tempfile.mktemp(suffix=".mp4")
    try:
        _run_inference(ref, audio_path, out_path,
                       req.inference_steps or 25, req.resolution or 512, req.seed, req.config_name)
        return {
            "video_base64": base64.b64encode(Path(out_path).read_bytes()).decode(),
            "duration": _probe_duration(out_path), "backend": _engine.mode,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in (audio_path, out_path):
            if os.path.exists(p):
                os.unlink(p)


@app.post("/api/avatar/register")
def register(req: RegisterRequest):
    """注册形象参考视频到 avatars_dir/<id>/reference.mp4"""
    try:
        avatar_dir = _avatars_dir / req.avatar_id
        avatar_dir.mkdir(parents=True, exist_ok=True)
        (avatar_dir / "reference.mp4").write_bytes(base64.b64decode(req.reference_video_base64))
        return {"success": True, "avatar_id": req.avatar_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="EnlyAI Avatar Worker (resident LatentSync)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--mode", default=None, choices=["resident", "subprocess", "placeholder"],
                        help="覆盖 LATENTSYNC_MODE（默认 resident，失败自动回退）")
    args = parser.parse_args()
    if args.mode:
        os.environ["LATENTSYNC_MODE"] = args.mode

    global _engine
    _engine = LatentSyncEngine()
    print(f"Avatar Worker 启动: http://{args.host}:{args.port} "
          f"(GPU={os.environ.get('CUDA_VISIBLE_DEVICES', 'default')})")
    _engine.load()   # 启动即加载一次，常驻
    print(f"就绪: mode={_engine.mode}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
