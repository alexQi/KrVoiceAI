"""TTS API 服务（云端 GPU 部署）

支持两种推理后端：
- CosyVoice2：阿里 FunAudioLLM 开源，零样本声音克隆，中文强（推荐，backend=cosyvoice）
- GPT-SoVITS：社区成熟声音克隆（备选，backend=gpt_sovits）

在 GPU 机器上启动此服务，提供 TTS 声音克隆 API。
本地 EnlyAI 通过 GPURunner 调用（provider=cosyvoice / gpt_sovits，二者共用同一 HTTP 契约）。

启动方式（用对应引擎环境的 python）：
    COSYVOICE_DIR=/path/to/CosyVoice \
    TTS_BACKEND=cosyvoice CUDA_VISIBLE_DEVICES=6 \
    python -m krvoiceai.api.tts_server --port 9880 --backend cosyvoice

依赖（云端安装，参考 scripts/setup_cloud_gpu.sh）：
    pip install fastapi uvicorn
    # CosyVoice2（推荐）：
    git clone https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice && git submodule update --init --recursive
    pip install -r requirements.txt
    # 下载权重到 pretrained_models/CosyVoice2-0.5B（modelscope/huggingface）
    # 关键环境变量：
    #   COSYVOICE_DIR         CosyVoice 仓库根（默认 ./CosyVoice）
    #   COSYVOICE_MODEL_DIR   权重目录（默认 <DIR>/pretrained_models/CosyVoice2-0.5B）
    #   COSYVOICE_FP16        是否 fp16（默认 true，GPU 用）
    # 或 GPT-SoVITS（备选）：pip install gpt-sovits
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import wave
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="EnlyAI TTS Server", version="2.0")

# 推理后端实例（延迟加载）
_tts_backend: Optional[Any] = None
_voices_dir = Path(os.environ.get("VOICES_DIR", "./config/voices"))
_backend_name = os.environ.get("TTS_BACKEND", "cosyvoice")  # cosyvoice / gpt_sovits


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "default"
    speed: float = 1.0


class RegisterVoiceRequest(BaseModel):
    voice_id: str
    sample_audio_base64: str
    # 可选：参考音频对应文本（提供后 CosyVoice2 用 zero_shot，质量更佳；
    # 不提供则用 cross_lingual，无需转写也能克隆）
    prompt_text: Optional[str] = None


# ────────────────────────── 后端加载 ──────────────────────────

def _get_tts_backend():
    """延迟加载推理后端。返回统一封装对象或 None（None→占位静音降级）。"""
    global _tts_backend
    if _tts_backend is not None:
        return _tts_backend

    if _backend_name == "cosyvoice":
        _tts_backend = _load_cosyvoice()
    elif _backend_name == "gpt_sovits":
        _tts_backend = _load_gpt_sovits()
    else:
        print(f"[warn] 未知 TTS 后端 {_backend_name}，使用占位静音降级")

    return _tts_backend


def _load_cosyvoice():
    """加载 CosyVoice2 后端

    CosyVoice2 是仓库形态（非 pip 包），需把仓库及 third_party/Matcha-TTS 加入 sys.path。
    参考：https://github.com/FunAudioLLM/CosyVoice
    """
    import sys
    repo_dir = Path(os.environ.get("COSYVOICE_DIR", "./CosyVoice")).resolve()
    model_dir = Path(os.environ.get(
        "COSYVOICE_MODEL_DIR", str(repo_dir / "pretrained_models" / "CosyVoice2-0.5B")
    ))
    fp16 = os.environ.get("COSYVOICE_FP16", "true").lower() in ("1", "true", "yes", "on")

    if not repo_dir.exists() or not model_dir.exists():
        print(f"[warn] CosyVoice2 未就绪 repo={repo_dir}(exists={repo_dir.exists()}) "
              f"model={model_dir}(exists={model_dir.exists()})，使用占位静音降级")
        print("[info] 安装：git clone https://github.com/FunAudioLLM/CosyVoice && "
              "下载 CosyVoice2-0.5B 权重；设置 COSYVOICE_DIR")
        return None

    try:
        # 仓库 + Matcha-TTS 子模块加入 import 路径
        for p in (str(repo_dir), str(repo_dir / "third_party" / "Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav

        model = CosyVoice2(str(model_dir), load_jit=False, load_trt=False, fp16=fp16)
        print(f"[info] CosyVoice2 后端加载成功 model={model_dir.name} "
              f"fp16={fp16} sr={model.sample_rate}")
        return _CosyVoiceWrapper(model, load_wav)
    except Exception as e:
        print(f"[warn] CosyVoice2 加载失败：{e}，使用占位静音降级")
        return None


class _CosyVoiceWrapper:
    """CosyVoice2 统一接口封装"""

    def __init__(self, model, load_wav):
        self.model = model
        self.load_wav = load_wav
        self.sample_rate = int(model.sample_rate)

    def synthesize(
        self, text: str, ref_audio: str,
        prompt_text: str | None = None, speed: float = 1.0,
    ) -> tuple[bytes, int]:
        """零样本声音克隆合成

        有 prompt_text → inference_zero_shot（质量更佳）；
        无 → inference_cross_lingual（无需转写也能克隆）。

        Returns: (wav_bytes, sample_rate)
        """
        import torch
        prompt_16k = self.load_wav(ref_audio, 16000)

        chunks = []
        if prompt_text:
            gen = self.model.inference_zero_shot(
                text, prompt_text, prompt_16k, stream=False, speed=speed,
            )
        else:
            gen = self.model.inference_cross_lingual(
                text, prompt_16k, stream=False, speed=speed,
            )
        for out in gen:
            chunks.append(out["tts_speech"])

        if not chunks:
            raise RuntimeError("CosyVoice2 未产出音频")
        speech = torch.cat(chunks, dim=1)  # [1, N] float32 in [-1, 1]
        return _tensor_to_wav_bytes(speech, self.sample_rate), self.sample_rate


def _load_gpt_sovits():
    """加载 GPT-SoVITS 后端（备选）"""
    try:
        from GPT_SoVITS.inference_webui import (  # noqa: F401
            change_gpt_weights, change_sovits_weights, get_tts_wav,
        )
        print("[info] GPT-SoVITS 后端加载成功")
        return _GptSovitsWrapper(get_tts_wav)
    except ImportError:
        print("[warn] GPT-SoVITS 未安装，使用占位静音降级")
        return None


class _GptSovitsWrapper:
    """GPT-SoVITS 统一接口封装（骨架，接口随版本调整）"""

    def __init__(self, get_tts_wav):
        self.get_tts_wav = get_tts_wav

    def synthesize(
        self, text: str, ref_audio: str,
        prompt_text: str | None = None, speed: float = 1.0,
    ) -> tuple[bytes, int]:
        # GPT-SoVITS 真实调用（接口随版本调整）：
        #   sr, audio = self.get_tts_wav(ref_wav_path=ref_audio, prompt_text=prompt_text,
        #                                text=text, speed=speed)
        #   return _pcm_to_wav_bytes(audio, sr), sr
        raise NotImplementedError("GPT-SoVITS 后端需按实际版本补全 synthesize()")


# ────────────────────────── 音频工具 ──────────────────────────

def _tensor_to_wav_bytes(speech, sample_rate: int) -> bytes:
    """torch float32 张量 [1, N]（[-1,1]）→ 16bit PCM wav bytes"""
    import numpy as np
    arr = speech.squeeze(0).detach().cpu().numpy()
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _silence_wav_bytes(text: str, sample_rate: int = 32000) -> tuple[bytes, float, int]:
    """占位：生成极低幅度噪声 wav（避免完全静音被解码器拒绝）"""
    import numpy as np
    duration = max(0.5, len(text) / 4.5)
    audio = (np.random.randn(int(duration * sample_rate)) * 0.001 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue(), duration, sample_rate


def _wav_duration(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


# ────────────────────────── 路由 ──────────────────────────

@app.get("/health")
def health():
    backend = _get_tts_backend()
    return {
        "status": "ok",
        "service": "tts",
        "backend": _backend_name,
        "backend_ready": backend is not None,
    }


@app.post("/api/tts/synthesize")
def synthesize(req: SynthesizeRequest):
    """文本转语音（声音克隆）"""
    try:
        # 查找音色参考音频（不依赖模型加载）
        voice_dir = _voices_dir / req.voice_id
        ref_audio = None
        for name in ("sample.wav", "sample.mp3", "ref.wav"):
            p = voice_dir / name
            if p.exists():
                ref_audio = p
                break
        if not ref_audio:
            raise HTTPException(status_code=404, detail=f"音色 {req.voice_id} 未注册")

        # 可选 prompt 文本（注册时保存）
        prompt_text = None
        prompt_file = voice_dir / "prompt.txt"
        if prompt_file.exists():
            prompt_text = prompt_file.read_text(encoding="utf-8").strip() or None

        backend = _get_tts_backend()
        if backend is not None:
            try:
                wav_bytes, sr = backend.synthesize(
                    text=req.text, ref_audio=str(ref_audio),
                    prompt_text=prompt_text, speed=req.speed,
                )
                return {
                    "audio_base64": base64.b64encode(wav_bytes).decode(),
                    "duration": _wav_duration(wav_bytes),
                    "sample_rate": sr,
                    "voice_id": req.voice_id,
                    "backend": _backend_name,
                }
            except Exception as e:
                print(f"[error] {_backend_name} 合成失败，降级占位：{e}")

        # 占位降级：静音 wav（实际部署时后端就绪即不会走到这里）
        wav_bytes, duration, sr = _silence_wav_bytes(req.text)
        return {
            "audio_base64": base64.b64encode(wav_bytes).decode(),
            "duration": duration,
            "sample_rate": sr,
            "voice_id": req.voice_id,
            "backend": "placeholder",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts/register_voice")
def register_voice(req: RegisterVoiceRequest):
    """注册音色（保存参考音频，可选保存 prompt 文本）"""
    try:
        voice_dir = _voices_dir / req.voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        audio_bytes = base64.b64decode(req.sample_audio_base64)
        (voice_dir / "sample.wav").write_bytes(audio_bytes)
        if req.prompt_text:
            (voice_dir / "prompt.txt").write_text(req.prompt_text, encoding="utf-8")
        return {"success": True, "voice_id": req.voice_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="EnlyAI TTS Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument(
        "--backend", default="cosyvoice",
        choices=["cosyvoice", "gpt_sovits"],
        help="推理后端（cosyvoice 推荐，gpt_sovits 备选）",
    )
    args = parser.parse_args()

    global _backend_name
    _backend_name = args.backend
    os.environ["TTS_BACKEND"] = args.backend

    # 预热后端（启动即加载，避免首个请求慢）
    backend = _get_tts_backend()
    status = "就绪" if backend is not None else "占位降级（未装推理依赖）"
    print(f"TTS 服务启动: http://{args.host}:{args.port}")
    print(f"后端: {args.backend} [{status}]")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
