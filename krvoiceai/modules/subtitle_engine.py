"""字幕生成模块

三种 provider：
- mimo:   调用小米 MiMo ASR API（OpenAI 兼容 chat/completions 端点）
- funasr: 调用 FunASR 服务（本地 HTTP API）进行语音识别 + 时间戳对齐
- mock:   优先复用 TTS 时间戳，否则按文本长度估算

输出：SRT 格式字幕文件
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import httpx

from ..core.audio_utils import estimate_speech_duration, split_text_to_segments
from ..core.base_module import BaseModule, JobContext, ModuleResult


def format_srt_time(seconds: float) -> str:
    """秒数转 SRT 时间格式 HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    # 用 round 避免浮点精度问题（如 3661.999 -> 998）
    ms = round((seconds % 1) * 1000)
    if ms >= 1000:  # 四舍五入进位
        ms = 0
        seconds += 1
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """将分句时间戳列表转为 SRT 字符串"""
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        text = seg["text"].strip()
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # 空行分隔
    return "\n".join(lines).rstrip("\n") + "\n"


class SubtitleEngine(BaseModule):
    """字幕生成模块"""

    name = "subtitle"
    requires_gpu = False  # FunASR CPU 也可跑

    def __init__(self, config=None):
        super().__init__(config)
        self.provider = self.config.get("asr.provider", "mock")
        self.model = self.config.get("asr.model", "paraformer-zh")
        self.language = self.config.get("asr.language", "zh")
        self.max_chars = self.config.get("asr.subtitle.max_chars_per_line", 18)
        # MiMo ASR 配置
        self.mimo_api_base = self.config.get("asr.api_base", "")
        self.mimo_api_key = self.config.get("asr.api_key", "")
        self.mimo_model = self.config.get("asr.mimo_model", "mimo-v2.5-asr")
        self.timeout = self.config.get("asr.timeout", 120)

    def setup(self) -> None:
        if self.provider == "mimo":
            if not self.mimo_api_key or not self.mimo_api_base:
                self.logger.warning(
                    "MiMo ASR 未配置 api_key/api_base，降级到 mock 模式"
                )
                self.provider = "mock"
            else:
                self.logger.info(f"MiMo ASR 模式 model={self.mimo_model}")
        elif self.provider == "funasr":
            # 检查 FunASR 是否可用（尝试 import）
            try:
                import funasr  # noqa: F401
                self._funasr_available = True
                self.logger.info("FunASR 本地可用")
            except ImportError:
                self._funasr_available = False
                self.logger.warning(
                    "FunASR 未安装，降级到 mock 模式（使用 TTS 时间戳）"
                )
                self.provider = "mock"
        else:
            self._funasr_available = False
        self.logger.info(f"字幕模块初始化 provider={self.provider}")
        super().setup()

    def run(self, ctx: JobContext) -> ModuleResult:
        """根据音频生成字幕"""
        if not ctx.audio_path or not ctx.audio_path.exists():
            return ModuleResult(success=False, error="无音频文件，无法生成字幕")

        output_path = ctx.work_dir / "subtitle.srt"

        try:
            if self.provider == "mimo":
                segments = self._recognize_mimo(ctx)
            elif self.provider == "funasr" and self._funasr_available:
                segments = self._recognize_funasr(ctx)
            else:
                segments = self._generate_mock(ctx)

            srt_content = segments_to_srt(segments)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(srt_content, encoding="utf-8")

            ctx.subtitle_path = output_path
            ctx.metadata["subtitle_segments"] = segments

            return ModuleResult(
                success=True,
                data={
                    "subtitle_path": str(output_path),
                    "segment_count": len(segments),
                    "provider": self.provider,
                },
            )
        except Exception as e:
            return ModuleResult(success=False, error=str(e))

    def _recognize_mimo(self, ctx: JobContext) -> list[dict]:
        """使用小米 MiMo ASR 识别音频

        MiMo ASR 特点：
        - 端点：{api_base}/chat/completions
        - 音频以 data URL 格式传入（data:audio/mp3;base64,...）
        - 不接受 text 部分（网关注入）
        - 返回识别文本在 choices[0].message.content
        - 不返回时间戳，需按文本长度估算
        """
        self.logger.info(f"MiMo ASR 识别音频: {ctx.audio_path}")

        # 读取音频并转 base64 data URL
        audio_path = ctx.audio_path
        audio_bytes = audio_path.read_bytes()
        # 判断格式
        ext = audio_path.suffix.lower().lstrip(".")
        mime = "audio/wav" if ext == "wav" else "audio/mp3"
        audio_b64 = base64.b64encode(audio_bytes).decode()
        data_url = f"data:{mime};base64,{audio_b64}"

        payload = {
            "model": self.mimo_model,
            "messages": [
                {"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": data_url, "format": ext or "mp3"}}
                ]}
            ],
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.mimo_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.mimo_api_base.rstrip('/')}/chat/completions"

        r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            self.logger.warning("MiMo ASR 返回空内容，降级到 mock")
            return self._generate_mock(ctx)

        self.logger.info(f"MiMo ASR 识别结果: {content[:100]}...")

        # MiMo ASR 不返回时间戳，按文本长度估算
        return self._split_text_by_duration(content, ctx.audio_duration)

    def _recognize_funasr(self, ctx: JobContext) -> list[dict]:
        """使用 FunASR 识别音频并生成带时间戳的分句"""
        self.logger.info(f"FunASR 识别音频: {ctx.audio_path}")
        from funasr import AutoModel

        model = AutoModel(
            model=self.model,
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True,
        )

        result = model.generate(
            input=str(ctx.audio_path),
            batch_size_s=300,
            sentence_timestamp=True,
        )

        segments: list[dict] = []
        for res in result:
            sentence_list = res.get("sentence_info", [])
            if sentence_list:
                for s in sentence_list:
                    text = s.get("text", "").strip()
                    if text:
                        # 长句切分
                        if len(text) > self.max_chars:
                            sub_segs = split_text_to_segments(text, self.max_chars)
                            total_dur = s.get("end", 0) - s.get("start", 0)
                            for j, sub in enumerate(sub_segs):
                                sub_start = s.get("start", 0) + j * total_dur / len(sub_segs)
                                sub_end = s.get("start", 0) + (j + 1) * total_dur / len(sub_segs)
                                segments.append({
                                    "text": sub,
                                    "start": round(sub_start / 1000, 3),
                                    "end": round(sub_end / 1000, 3),
                                })
                        else:
                            segments.append({
                                "text": text,
                                "start": round(s.get("start", 0) / 1000, 3),
                                "end": round(s.get("end", 0) / 1000, 3),
                            })
            else:
                # 无 sentence_info，用纯文本
                text = res.get("text", "").strip()
                if text:
                    segments.extend(self._split_text_by_duration(
                        text, ctx.audio_duration
                    ))

        self.logger.info(f"FunASR 识别完成，{len(segments)} 条字幕")
        return segments

    def _generate_mock(self, ctx: JobContext) -> list[dict]:
        """Mock 模式：优先复用 TTS 时间戳，否则按文本估算"""
        # 优先使用 TTS 模块生成的时间戳
        tts_ts = ctx.metadata.get("tts_timestamps")
        if tts_ts:
            self.logger.info(f"复用 TTS 时间戳生成字幕，{len(tts_ts)} 条")
            # 按最大字数切分过长的段
            segments: list[dict] = []
            for ts in tts_ts:
                text = ts["text"]
                if len(text) > self.max_chars:
                    sub_segs = split_text_to_segments(text, self.max_chars)
                    dur = ts["end"] - ts["start"]
                    for j, sub in enumerate(sub_segs):
                        s = ts["start"] + j * dur / len(sub_segs)
                        e = ts["start"] + (j + 1) * dur / len(sub_segs)
                        segments.append({
                            "text": sub,
                            "start": round(s, 3),
                            "end": round(e, 3),
                        })
                else:
                    segments.append(ts)
            return segments

        # 否则按文案文本估算
        text = ctx.script_text or ctx.input_script
        if not text:
            return [{
                "text": "（无文案）",
                "start": 0.0,
                "end": ctx.audio_duration,
            }]

        self.logger.info("按文本长度估算字幕时间戳")
        return self._split_text_by_duration(text, ctx.audio_duration)

    def _split_text_by_duration(
        self, text: str, total_duration: float
    ) -> list[dict]:
        """按文本切分并按字数比例分配时长"""
        segments = split_text_to_segments(text, self.max_chars)
        total_chars = sum(len(s) for s in segments) or 1
        result: list[dict] = []
        offset = 0.0
        for seg in segments:
            seg_dur = total_duration * len(seg) / total_chars
            result.append({
                "text": seg,
                "start": round(offset, 3),
                "end": round(offset + seg_dur, 3),
            })
            offset += seg_dur
        return result
