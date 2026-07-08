"""单条视频多卡分片：把 TTS 音频按句子边界规划成 K 段（M2）

复用 TTS 段级时间戳（句间有停顿静音，天然是干净切点），贪心均衡各段时长，
保证最小段时长（LatentSync 太短段效率低）。每段映射到参考视频的时间窗口，
供各 worker 并行推理，再按序拼接。
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def plan_segments(
    timestamps: list[dict],
    audio_duration: float,
    n_workers: int,
    min_seg_sec: float = 2.5,
    target_segments: int | None = None,
) -> list[dict]:
    """按句子边界把音频规划成均衡的 K 段。

    Args:
        timestamps: TTS 段级时间戳 [{"text","start","end"}, ...]
        audio_duration: 音频总时长
        n_workers: 可用 worker 数（默认目标段数）
        min_seg_sec: 最小段时长
        target_segments: 目标段数（默认 = n_workers）

    Returns:
        [{"seg_index","t_start","t_end","ref_start","ref_end","text"}, ...]
    """
    total = float(audio_duration or 0)
    if not timestamps:
        total = total or 1.0
        return [{"seg_index": 0, "t_start": 0.0, "t_end": total,
                 "ref_start": 0.0, "ref_end": total, "text": ""}]

    sents = [{"start": float(t["start"]), "end": float(t["end"]), "text": t.get("text", "")}
             for t in timestamps]
    total = total or sents[-1]["end"]
    K = target_segments or n_workers
    K = max(1, min(int(K), len(sents)))
    target_dur = total / K if K else total

    shards: list[dict] = []
    cur: dict | None = None
    for s in sents:
        if cur is None:
            cur = {"t_start": s["start"], "t_end": s["end"], "text": s["text"]}
            continue
        cur_dur = cur["t_end"] - cur["t_start"]
        # 当前段已达目标时长且满足最小时长，且还需要更多段 → 封段、开新段
        if cur_dur >= target_dur and cur_dur >= min_seg_sec and len(shards) < K - 1:
            shards.append(cur)
            cur = {"t_start": s["start"], "t_end": s["end"], "text": s["text"]}
        else:
            cur["t_end"] = s["end"]
            cur["text"] += s["text"]
    if cur is not None:
        shards.append(cur)

    # 合并过短的尾段到前一段
    if len(shards) >= 2 and (shards[-1]["t_end"] - shards[-1]["t_start"]) < min_seg_sec:
        shards[-2]["t_end"] = shards[-1]["t_end"]
        shards[-2]["text"] += shards[-1]["text"]
        shards.pop()

    for i, sh in enumerate(shards):
        # 参考视频窗口 = 音频段窗口（reference 比音频短时，worker 侧 LatentSync 会循环参考）
        sh["seg_index"] = i
        sh["ref_start"] = sh["t_start"]
        sh["ref_end"] = sh["t_end"]
    return shards


def slice_audio(src: Path, t_start: float, t_end: float, out: Path) -> Path:
    """按 [t_start, t_end] 切音频（重编码为 pcm wav，保证切点准确）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.05, t_end - t_start)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t_start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
         "-c:a", "pcm_s16le", "-ar", "16000", str(out)],
        capture_output=True, check=True,
    )
    return out
