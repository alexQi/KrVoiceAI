#!/usr/bin/env python3
"""M1 接缝画质验证：把一段音频从切点分成两段，各自调 worker 的 /generate_segment，
拼接后用整条音频覆盖，产出成片供**肉眼检查切点处的接缝**。

这是多卡分片方案最大风险（接缝伪影）的最小验证工具。切点应落在句子/静音处。

前置：worker 已启动，且形象已注册（reference.mp4 在 worker 的 avatars_dir/<avatar_id>/）。

用法：
    python scripts/validate_segment_seam.py \
        --worker http://127.0.0.1:8010 \
        --avatar anchor_wang \
        --audio /path/to/tts_audio.wav \
        --split-at 6.5 \
        --out seam_test.mp4

然后播放 seam_test.mp4，重点看第 6.5 秒附近嘴型/头部是否平滑。
"""
import argparse
import base64
import subprocess
import tempfile
from pathlib import Path

import requests


def probe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def slice_audio(src: str, start: float, end: float, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", src, "-t", f"{end - start:.3f}",
         "-c:a", "pcm_s16le", out],
        capture_output=True, check=True,
    )


def gen_segment(worker: str, avatar: str, audio_wav: str, ref_start: float, ref_end: float,
                seg_index: int, steps: int, resolution: int, seed: int) -> bytes:
    audio_b64 = base64.b64encode(Path(audio_wav).read_bytes()).decode()
    resp = requests.post(
        f"{worker}/api/avatar/generate_segment",
        json={
            "audio_base64": audio_b64, "avatar_id": avatar,
            "ref_start": ref_start, "ref_end": ref_end, "seg_index": seg_index,
            "inference_steps": steps, "resolution": resolution, "seed": seed,
        },
        timeout=1800,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"  段 {seg_index}: mode={data.get('mode')} duration={data.get('duration'):.2f}s")
    return base64.b64decode(data["video_base64"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="http://127.0.0.1:8010")
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--audio", required=True, help="整条 TTS 音频 wav")
    ap.add_argument("--split-at", type=float, required=True, help="切点（秒，建议落在句子/静音处）")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--out", default="seam_test.mp4")
    args = ap.parse_args()

    dur = probe_duration(args.audio)
    split = args.split_at
    assert 0 < split < dur, f"切点 {split} 应在 (0, {dur:.2f}) 内"
    print(f"音频总长 {dur:.2f}s，切点 {split:.2f}s → 段A[0,{split:.2f}] 段B[{split:.2f},{dur:.2f}]")

    tmp = Path(tempfile.mkdtemp())
    a_wav, b_wav = str(tmp / "a.wav"), str(tmp / "b.wav")
    slice_audio(args.audio, 0, split, a_wav)
    slice_audio(args.audio, split, dur, b_wav)

    print("并行段推理（顺序调用同一 worker；多 worker 由 M2 调度器并发）：")
    a_vid = gen_segment(args.worker, args.avatar, a_wav, 0, split, 0, args.steps, args.resolution, args.seed)
    b_vid = gen_segment(args.worker, args.avatar, b_wav, split, dur, 1, args.steps, args.resolution, args.seed)

    a_mp4, b_mp4 = str(tmp / "a.mp4"), str(tmp / "b.mp4")
    Path(a_mp4).write_bytes(a_vid)
    Path(b_mp4).write_bytes(b_vid)

    # 拼接两段视频（重编码统一参数）+ 整条音频覆盖
    concat_list = tmp / "list.txt"
    concat_list.write_text(f"file '{Path(a_mp4).absolute()}'\nfile '{Path(b_mp4).absolute()}'\n")
    concat_mp4 = str(tmp / "concat.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", concat_mp4],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", concat_mp4, "-i", args.audio,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", args.out],
        capture_output=True, check=True,
    )
    print(f"\n✅ 成片：{args.out} —— 重点检查第 {split:.2f}s 附近的接缝（嘴型/头部是否平滑）")


if __name__ == "__main__":
    main()
