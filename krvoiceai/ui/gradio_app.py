"""Gradio Web UI - KrVoiceAI 虚拟人口播智能体

界面结构（7 个标签页）：
  Tab 1: 🎬 一键生成     - 全流程一键产出（文案→语音→数字人→字幕→成片）
  Tab 2: 🎙️ 声音克隆     - 上传音频样本，注册零样本克隆音色（MOSS-TTS-Nano）
  Tab 3: 🧑 形象管理     - 上传口播人像视频，注册 Wav2Lip 数字人形象
  Tab 4: 🎞️ 画中画编辑器 - 可视化时间线：在指定时间段插入插播画面（cut 全屏替换 / pip 角窗）
  Tab 5: 📤 多平台发布   - 生成发布清单 + 半自动打开抖音/B站/快手创作者中心预填
  Tab 6: ⚙️ 设置         - TTS / 数字人 / GFPGAN / 字幕 / LLM / 发布 等全部配置项
  Tab 7: 📋 任务管理     - 历史任务、断点续跑、删除
"""
from __future__ import annotations

import json
import shutil
import time
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional

try:
    import gradio as gr
except ImportError:
    gr = None

from ..app import KrVoiceAI
from ..core.settings_manager import get_settings_manager
from ..core.logger import get_logger

# 步骤中文名映射
STEP_NAMES = {
    "script_extract": "文案提取",
    "script_write": "文案仿写",
    "originality_check": "原创检测",
    "tts": "语音合成",
    "avatar": "数字人生成",
    "subtitle": "字幕生成",
    "broll": "画中画/插播",
    "compose": "视频合成",
    "title": "标题生成",
    "cover": "封面生成",
    "publish": "多平台发布",
}

STEP_ORDER = [
    "script_extract", "script_write", "originality_check", "tts", "avatar",
    "subtitle", "broll", "compose", "title", "cover", "publish",
]

STATUS_ICON = {
    "pending": "⏳", "running": "🔄", "success": "✅",
    "failed": "❌", "skipped": "⏭️", "retry": "🔁",
}

# 各平台创作者发布页 URL
PUBLISH_URLS = {
    "douyin": "https://creator.douyin.com/creator-micro/content/upload",
    "bilibili": "https://member.bilibili.com/platform/upload/video/frame",
    "kuaishou": "https://cp.kuaishou.com/article/publish/video",
    "wechat_video": "https://channels.weixin.qq.com/platform/post/create",
}

_app: Optional[KrVoiceAI] = None

# 自定义 CSS（Gradio 6.0+ 通过 launch(css=...) 注入）
CUSTOM_CSS = """
.step-progress {
    font-family: monospace; font-size: 14px; line-height: 2;
    padding: 16px; background: #f7f7f8; border-radius: 8px;
    border: 1px solid #e0e0e0;
}
.section-title {
    font-size: 18px; font-weight: bold; color: #2563eb;
    margin: 12px 0 8px 0; padding-bottom: 6px; border-bottom: 2px solid #2563eb;
}
"""


def _get_app() -> KrVoiceAI:
    global _app
    if _app is None:
        _app = KrVoiceAI()
    return _app


def _format_progress(steps_state: dict) -> str:
    lines = []
    for step in STEP_ORDER:
        name = STEP_NAMES.get(step, step)
        status = steps_state.get(step, "pending")
        icon = STATUS_ICON.get(status, "⏳")
        lines.append(f"{icon} {name}")
    return "\n".join(lines)


def _build_ui() -> "gr.Blocks":
    """构建 Gradio 界面"""
    app = _get_app()

    with gr.Blocks(title="KrVoiceAI 虚拟人口播智能体") as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 16px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 16px;">
            <h1 style="color: white; margin: 0; font-size: 28px;">KrVoiceAI 虚拟人口播智能体</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0;">本地 CPU 声音克隆 · Wav2Lip 数字人 · 画中画时间线 · 一键多平台发布</p>
        </div>
        """)

        # ============ Tab 1: 一键生成 ============
        with gr.Tab("🎬 一键生成"):
            gr.HTML('<div class="section-title">输入设置</div>')
            with gr.Row():
                with gr.Column(scale=3):
                    script_input = gr.Textbox(
                        label="口播文案", lines=6,
                        placeholder="直接输入文案，或留空填参考链接自动提取...",
                        info="支持 150-500 字，系统会自动润色/仿写",
                    )
                    ref_url = gr.Textbox(
                        label="参考视频链接（可选）",
                        placeholder="粘贴抖音/快手/B站链接，自动提取文案",
                    )
                    with gr.Row():
                        mode_dd = gr.Dropdown(
                            label="文案模式",
                            choices=[("润色", "polish"), ("仿写", "rewrite"), ("全新生成", "generate")],
                            value="polish",
                        )
                        platform_dd = gr.Dropdown(
                            label="目标平台",
                            choices=["douyin", "bilibili", "kuaishou", "wechat_video"],
                            value="douyin",
                        )
                    with gr.Row():
                        avatar_dd = gr.Dropdown(
                            label="数字人形象", choices=["default"], value="default",
                            allow_custom_value=True,
                        )
                        voice_dd = gr.Dropdown(
                            label="音色", choices=["default"], value="default",
                            allow_custom_value=True,
                        )
                    with gr.Row():
                        auto_publish = gr.Checkbox(label="生成后自动发布", value=False)
                        broll_state = gr.State([])  # 画中画片段（从编辑器带入）
                        refresh_btn = gr.Button("🔄 刷新形象/音色", size="sm")

                    run_btn = gr.Button("🚀 开始生成视频", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.HTML('<div class="section-title">实时进度</div>')
                    progress_out = gr.Textbox(
                        label="流水线进度", value=_format_progress({}),
                        elem_classes=["step-progress"], lines=12, interactive=False,
                    )
                    status_out = gr.Textbox(label="任务状态", lines=2, interactive=False)

            gr.HTML('<div class="section-title">生成结果</div>')
            with gr.Row():
                with gr.Column(scale=2):
                    video_out = gr.Video(label="成片预览")
                with gr.Column(scale=1):
                    title_out = gr.Textbox(label="标题", interactive=False)
                    cover_out = gr.Image(label="封面", type="filepath")
                    script_out = gr.Textbox(label="最终文案", lines=5, interactive=False)
                    info_out = gr.JSON(label="详细信息")
                    with gr.Row():
                        publish_douyin_btn = gr.Button("📤 发布到抖音", size="sm")
                        publish_bili_btn = gr.Button("📤 发布到B站", size="sm")
                    final_video_state = gr.State(None)

            def _run(script, url, avatar, voice, mode, platform, publish, broll):
                steps_state = {s: "pending" for s in STEP_ORDER}
                def progress_cb(step_name, status, data):
                    steps_state[step_name] = status
                result = app.submit_and_run(
                    script=script, reference_video_url=url or None,
                    avatar_id=avatar, voice_id=voice, script_mode=mode,
                    platform=platform, auto_publish=publish,
                    broll_clips=broll or None, progress_callback=progress_cb,
                )
                progress_text = _format_progress(steps_state)
                status_text = f"任务 {result['job_id']}: {result['status']}"
                if result.get("error"):
                    status_text += f" | 错误: {result['error']}"
                output = result.get("output", {})
                return (
                    progress_text, status_text,
                    output.get("final_video"), output.get("title", ""),
                    output.get("cover"), output.get("script_text", ""), result,
                    output.get("final_video"),
                )

            def _refresh():
                avatars = app.list_avatars()
                voices = app.list_voices()
                a_ids = [a["avatar_id"] for a in avatars] or ["default"]
                v_ids = [v["voice_id"] for v in voices] or ["default"]
                return (
                    gr.update(choices=a_ids, value=a_ids[0]),
                    gr.update(choices=v_ids, value=v_ids[0]),
                )

            run_btn.click(
                _run,
                inputs=[script_input, ref_url, avatar_dd, voice_dd,
                        mode_dd, platform_dd, auto_publish, broll_state],
                outputs=[progress_out, status_out, video_out, title_out,
                         cover_out, script_out, info_out, final_video_state],
            )
            refresh_btn.click(_refresh, outputs=[avatar_dd, voice_dd])
            demo.load(_refresh, outputs=[avatar_dd, voice_dd])

            def _open_publish(platform, video_path):
                if not video_path:
                    return f"⚠️ 请先生成视频，再发布到 {platform}"
                url = PUBLISH_URLS.get(platform)
                if url:
                    try:
                        webbrowser.open(url)
                        return f"✅ 已打开 {platform} 创作者发布页。\n请在浏览器中上传视频文件：\n{video_path}"
                    except Exception as e:
                        return f"⚠️ 无法打开浏览器: {e}\n请手动访问: {url}\n视频: {video_path}"
                return f"⚠️ 不支持的平台: {platform}"

            publish_douyin_btn.click(
                lambda v: _open_publish("douyin", v),
                inputs=[final_video_state], outputs=[status_out],
            )
            publish_bili_btn.click(
                lambda v: _open_publish("bilibili", v),
                inputs=[final_video_state], outputs=[status_out],
            )

        # ============ Tab 2: 声音克隆 ============
        with gr.Tab("🎙️ 声音克隆"):
            gr.Markdown("### 🎙️ 零样本声音克隆（MOSS-TTS-Nano）")
            gr.Markdown(
                "上传 **5-30 秒** 干净人声音频（wav/mp3），系统提取音色特征。"
                "之后即可用该音色合成任意文案。本地 CPU 运行，无需上传云端。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    voice_id_input = gr.Textbox(
                        label="音色名称（英文/拼音，如 teacher_li）",
                        placeholder="例如：teacher_li",
                    )
                    voice_sample = gr.Audio(
                        label="上传参考音频（5-30秒人声）", type="filepath",
                    )
                    reg_voice_btn = gr.Button("💾 注册音色", variant="primary")
                    reg_voice_out = gr.Textbox(label="注册结果", interactive=False)

                    gr.HTML('<div class="section-title">已注册音色</div>')
                    voice_list_out = gr.JSON(label="音色列表", value=[])

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">试听克隆效果</div>')
                    test_voice_dd = gr.Dropdown(
                        label="选择音色", choices=["default"], value="default",
                        allow_custom_value=True,
                    )
                    test_text = gr.Textbox(
                        label="试听文案", lines=3,
                        value="你好，这是我的克隆声音，欢迎收听今天的口播内容。",
                    )
                    test_synth_btn = gr.Button("🎵 试听合成", variant="primary")
                    test_audio_out = gr.Audio(label="合成结果", type="filepath")
                    test_info_out = gr.Textbox(label="合成信息", interactive=False)

            def _register_voice(vid, sample):
                if not vid or not sample:
                    return "❌ 请填写音色名称并上传音频", None, gr.update()
                ok = app.register_voice(vid, Path(sample))
                voices = app.list_voices()
                v_ids = [v["voice_id"] for v in voices] or ["default"]
                msg = f"✅ 音色 {vid} 注册成功！" if ok else f"❌ 注册失败"
                return msg, voices, gr.update(choices=v_ids, value=vid)

            def _list_voices():
                return app.list_voices()

            def _test_synth(voice, text):
                if not text.strip():
                    return None, "❌ 请输入试听文案"
                try:
                    engine = app.modules.get("tts")
                    if engine is None:
                        return None, "❌ TTS 引擎未初始化"
                    with tempfile.TemporaryDirectory() as td:
                        out = Path(td) / "preview.wav"
                        if engine.provider == "moss_nano":
                            audio_path, dur, _ = engine._synth_moss_nano(text, voice, out)
                        else:
                            res = app.run_single_module("tts", script=text, voice_id=voice)
                            if not res.get("success"):
                                return None, f"❌ {res.get('error')}"
                            audio_path = Path(res["context"].get("audio_path"))
                            dur = res["context"].get("audio_duration", 0)
                        persist = Path("output") / f"voice_preview_{int(time.time())}.wav"
                        persist.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(audio_path, persist)
                    return str(persist), f"✅ 合成成功，时长 {dur:.1f}s，音色 {voice}"
                except Exception as e:
                    return None, f"❌ 合成失败: {e}"

            reg_voice_btn.click(
                _register_voice, inputs=[voice_id_input, voice_sample],
                outputs=[reg_voice_out, voice_list_out, test_voice_dd],
            )
            demo.load(_list_voices, outputs=[voice_list_out])
            test_synth_btn.click(
                _test_synth, inputs=[test_voice_dd, test_text],
                outputs=[test_audio_out, test_info_out],
            )

        # ============ Tab 3: 形象管理 ============
        with gr.Tab("🧑 形象管理"):
            gr.Markdown("### 🧑 数字人形象注册（Wav2Lip 视频驱动）")
            gr.Markdown(
                "上传一段 **正脸口播视频**（5-30秒，嘴部清晰，分辨率≥480p）。"
                "系统将保留原视频的头动/表情，仅替换嘴形对齐 TTS 语音。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    avatar_id_input = gr.Textbox(
                        label="形象名称（英文/拼音，如 anchor_wang）",
                        placeholder="例如：anchor_wang",
                    )
                    avatar_video = gr.Video(label="上传口播人像视频", include_audio=True)
                    reg_avatar_btn = gr.Button("💾 注册形象", variant="primary")
                    reg_avatar_out = gr.Textbox(label="注册结果", interactive=False)

                    gr.HTML('<div class="section-title">已注册形象</div>')
                    avatar_list_out = gr.JSON(label="形象列表", value=[])

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">形象预览</div>')
                    avatar_preview_dd = gr.Dropdown(
                        label="选择形象预览", choices=["default"], value="default",
                        allow_custom_value=True,
                    )
                    avatar_preview_video = gr.Video(label="参考视频")

            def _register_avatar(aid, video):
                if not aid or not video:
                    return "❌ 请填写形象名称并上传视频", None, gr.update()
                ok = app.register_avatar(aid, Path(video))
                avatars = app.list_avatars()
                a_ids = [a["avatar_id"] for a in avatars] or ["default"]
                msg = f"✅ 形象 {aid} 注册成功！" if ok else f"❌ 注册失败"
                return msg, avatars, gr.update(choices=a_ids, value=aid)

            def _list_avatars():
                return app.list_avatars()

            def _preview_avatar(aid):
                avatars_dir = Path(app.config.get("avatar.avatars_dir", "./config/avatars"))
                ref = avatars_dir / aid / "reference_video.mp4"
                if ref.exists():
                    return str(ref)
                return None

            reg_avatar_btn.click(
                _register_avatar, inputs=[avatar_id_input, avatar_video],
                outputs=[reg_avatar_out, avatar_list_out, avatar_preview_dd],
            )
            demo.load(_list_avatars, outputs=[avatar_list_out])
            avatar_preview_dd.change(
                _preview_avatar, inputs=[avatar_preview_dd],
                outputs=[avatar_preview_video],
            )

        # ============ Tab 4: 画中画/插播编辑器 ============
        with gr.Tab("🎞️ 画中画编辑器"):
            gr.Markdown("### 🎞️ 插播画面时间线编辑器")
            gr.Markdown(
                "在指定时间段插入 **插播画面**：\n"
                "- **cut（全屏替换）**：该时间段整屏播放插播视频（对标旗博士）\n"
                "- **pip（角窗画中画）**：插播视频以小窗叠加在右下角\n\n"
                "编辑完成后片段会在「一键生成」运行时自动应用。"
            )
            broll_table_state = gr.State([])

            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML('<div class="section-title">添加插播片段</div>')
                    broll_path = gr.Textbox(
                        label="插播视频/图片路径",
                        placeholder="例如：D:/videos/broll_intro.mp4",
                    )
                    with gr.Row():
                        broll_start = gr.Number(label="开始时间(秒)", value=0, minimum=0)
                        broll_end = gr.Number(label="结束时间(秒)", value=5, minimum=0)
                        broll_mode = gr.Dropdown(
                            label="模式", choices=[("全屏替换", "cut"), ("角窗画中画", "pip")],
                            value="cut",
                        )
                    add_broll_btn = gr.Button("➕ 添加片段", variant="primary")

                    gr.HTML('<div class="section-title">已添加片段</div>')
                    broll_df = gr.Dataframe(
                        headers=["路径", "开始(s)", "结束(s)", "模式"],
                        datatype=["str", "number", "number", "str"],
                        value=[], interactive=False, wrap=True,
                    )
                    with gr.Row():
                        clear_broll_btn = gr.Button("🗑️ 清空全部", size="sm")
                        del_idx = gr.Number(label="删除第几行(从0)", value=0, precision=0)
                        del_broll_btn = gr.Button("❌ 删除该行", size="sm")

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">时间线预览</div>')
                    timeline_out = gr.Textbox(
                        label="时间线", lines=10, interactive=False,
                        placeholder="添加片段后显示时间线...",
                    )
                    apply_broll_out = gr.Textbox(label="操作结果", interactive=False)

            def _df_from(clips):
                return [[c["path"], c["start"], c["end"], c["mode"]] for c in clips]

            def _timeline_from(clips):
                if not clips:
                    return "（空）"
                max_t = max(c["end"] for c in clips)
                width = 50
                lines = [f"总时长: {max_t:.1f}s"]
                for c in sorted(clips, key=lambda x: x["start"]):
                    if max_t <= 0:
                        bar = ""
                    else:
                        pos = int(c["start"] / max_t * width)
                        length = max(1, int((c["end"] - c["start"]) / max_t * width))
                        bar = " " * pos + ("█" * length)
                    tag = "全屏" if c["mode"] == "cut" else "角窗"
                    lines.append(f"[{c['start']:5.1f}-{c['end']:5.1f}] {tag} {bar}")
                return "\n".join(lines)

            def _add_clip(path, start, end, mode, current):
                if not path or not Path(path).exists():
                    return current, _df_from(current), _timeline_from(current), "❌ 路径无效或文件不存在"
                clip = {
                    "path": path, "start": float(start), "end": float(end),
                    "mode": mode, "volume": 1.0,
                }
                current = list(current) + [clip]
                return current, _df_from(current), _timeline_from(current), f"✅ 已添加，共 {len(current)} 段"

            def _clear():
                return [], [], "（空）", "已清空"

            def _del(idx, current):
                idx = int(idx)
                if 0 <= idx < len(current):
                    current = list(current)
                    current.pop(idx)
                return current, _df_from(current), _timeline_from(current), f"已删除第{idx}行"

            add_broll_btn.click(
                _add_clip,
                inputs=[broll_path, broll_start, broll_end, broll_mode, broll_table_state],
                outputs=[broll_table_state, broll_df, timeline_out, apply_broll_out],
            )
            clear_broll_btn.click(
                _clear,
                outputs=[broll_table_state, broll_df, timeline_out, apply_broll_out],
            )
            del_broll_btn.click(
                _del, inputs=[del_idx, broll_table_state],
                outputs=[broll_table_state, broll_df, timeline_out, apply_broll_out],
            )

        # ============ Tab 5: 多平台发布 ============
        with gr.Tab("📤 多平台发布"):
            gr.Markdown("### 📤 多平台发布")
            gr.Markdown(
                "支持 **抖音 / B站 / 快手 / 视频号**。\n\n"
                "**半自动模式（推荐，合规且规避风控）**：点击下方按钮，自动打开对应平台创作者中心，"
                "标题/文案/标签已生成在发布清单中，您只需在浏览器粘贴并上传视频，全程人工把关。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    pub_video = gr.Textbox(
                        label="视频文件路径",
                        placeholder="成片路径，或点击下方从最近任务载入",
                    )
                    load_latest_btn = gr.Button("📂 载入最近成功任务", size="sm")
                    pub_title = gr.Textbox(label="标题", placeholder="视频标题")
                    pub_desc = gr.Textbox(label="描述/文案", lines=3)
                    pub_tags = gr.Textbox(label="标签（逗号分隔）", placeholder="AI,数字人,口播")
                    gen_manifest_btn = gr.Button("📝 生成发布清单", variant="primary")
                    manifest_out = gr.JSON(label="发布清单", value=None)

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">一键打开平台发布页</div>')
                    open_douyin = gr.Button("🎬 抖音创作者中心", size="lg")
                    open_bili = gr.Button("📺 B站投稿", size="lg")
                    open_kuaishou = gr.Button("快手创作者", size="lg")
                    open_wechat = gr.Button("微信视频号", size="lg")
                    pub_status = gr.Textbox(label="操作结果", lines=3, interactive=False)
                    pub_guide = gr.Textbox(
                        label="发布清单内容（复制到平台）", lines=8, interactive=False,
                    )

            def _load_latest():
                jobs = app.list_jobs(limit=20)
                for j in jobs:
                    out = (j.get("output") or {})
                    fv = out.get("final_video")
                    if fv and Path(fv).exists():
                        return (
                            fv, out.get("title", ""),
                            (out.get("script_text", "") or "")[:200],
                        )
                return "", "", ""

            def _gen_manifest(video, title, desc, tags):
                if not video or not Path(video).exists():
                    return None, "❌ 视频文件不存在", ""
                tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
                manifest = {
                    "video_path": video, "title": title or "口播视频",
                    "description": desc, "tags": tag_list,
                    "platforms": list(PUBLISH_URLS.keys()),
                }
                guide = (
                    f"【标题】{manifest['title']}\n\n"
                    f"【描述】{manifest['description']}\n\n"
                    f"【标签】{' '.join('#' + t for t in tag_list)}\n\n"
                    f"【视频文件】{video}"
                )
                mf_path = Path("output") / f"publish_manifest_{int(time.time())}.json"
                mf_path.parent.mkdir(parents=True, exist_ok=True)
                mf_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                return manifest, f"✅ 清单已生成: {mf_path}", guide

            def _open(platform, video, title, desc, tags):
                url = PUBLISH_URLS.get(platform)
                if not url:
                    return f"⚠️ 不支持的平台: {platform}"
                try:
                    webbrowser.open(url)
                    return (f"✅ 已打开 {platform} 创作者中心。\n"
                            f"请上传视频并粘贴清单内容。\n\n视频文件：{video}")
                except Exception as e:
                    return f"⚠️ 浏览器打开失败: {e}\n请手动访问: {url}"

            load_latest_btn.click(
                _load_latest, outputs=[pub_video, pub_title, pub_desc],
            )
            gen_manifest_btn.click(
                _gen_manifest, inputs=[pub_video, pub_title, pub_desc, pub_tags],
                outputs=[manifest_out, pub_status, pub_guide],
            )
            for btn, plat in [
                (open_douyin, "douyin"), (open_bili, "bilibili"),
                (open_kuaishou, "kuaishou"), (open_wechat, "wechat_video"),
            ]:
                btn.click(
                    lambda v, t, d, g, p=plat: _open(p, v, t, d, g),
                    inputs=[pub_video, pub_title, pub_desc, pub_tags],
                    outputs=[pub_status],
                )

        # ============ Tab 6: 设置 ============
        with gr.Tab("⚙️ 设置"):
            settings_mgr = get_settings_manager()
            cur_tts = settings_mgr.get_section("tts", mask_sensitive=False) or {}
            cur_avatar = settings_mgr.get_section("avatar", mask_sensitive=False) or {}
            cur_sub = settings_mgr.get_section("subtitle", mask_sensitive=False) or {}
            cur_llm = settings_mgr.get_section("llm", mask_sensitive=False) or {}
            cur_pub = settings_mgr.get_section("publisher", mask_sensitive=False) or {}
            cur_moss = (cur_tts.get("moss_nano") or {})

            gr.HTML('<div class="section-title">🎤 语音合成（TTS）</div>')
            with gr.Row():
                tts_provider = gr.Dropdown(
                    label="TTS 引擎",
                    choices=[("MOSS-TTS-Nano 本地克隆", "moss_nano"),
                             ("小米 MiMo 云端", "mimo"),
                             ("GPT-SoVITS 云端", "gpt_sovits"),
                             ("Edge-TTS（无需GPU）", "edge_tts"),
                             ("Mock 测试", "mock")],
                    value=cur_tts.get("provider", "moss_nano"),
                )
                tts_threads = gr.Slider(
                    label="CPU 线程数", minimum=1, maximum=16, step=1,
                    value=cur_moss.get("cpu_threads", 4),
                )
                moss_builtin = gr.Dropdown(
                    label="内置音色（无克隆样本时）",
                    choices=["Junhao", "Trump", "Ava", "Bella", "Adam", "Nathan"],
                    value=cur_moss.get("builtin_voice", "Junhao"),
                )

            gr.HTML('<div class="section-title">🧑 数字人（Wav2Lip）</div>')
            with gr.Row():
                w2l = cur_avatar.get("wav2lip") or {}
                wav2lip_python = gr.Textbox(
                    label="Wav2Lip Python 解释器路径",
                    value=w2l.get("env_python",
                                  "D:/cursor_project/koubo/wav2lip_env/Scripts/python.exe"),
                    info="Python 3.8 venv 路径（含 torch）",
                )
            with gr.Row():
                gf = cur_avatar.get("gfpgan") or {}
                gfpgan_enabled = gr.Checkbox(
                    label="开启 GFPGAN 人脸增强（更清晰，但可能轻微跳帧）",
                    value=gf.get("enabled", False),
                )
                gfpgan_weight = gr.Textbox(
                    label="GFPGAN 权重路径", value=gf.get("weight_path", ""),
                )
                gfpgan_stride = gr.Slider(
                    label="GFPGAN 步长（1=逐帧最稳，4=最快易跳帧）",
                    minimum=1, maximum=8, step=1, value=gf.get("stride", 1),
                )

            gr.HTML('<div class="section-title">📝 字幕</div>')
            with gr.Row():
                sub_font = gr.Textbox(label="字体路径", value=cur_sub.get("font_path", ""))
                sub_size = gr.Slider(
                    label="字号", minimum=20, maximum=80, step=1,
                    value=cur_sub.get("font_size", 36),
                )
                sub_margin = gr.Slider(
                    label="底部边距", minimum=40, maximum=300, step=5,
                    value=cur_sub.get("margin_v", 120),
                )
                sub_karaoke = gr.Checkbox(
                    label="卡拉OK逐字高亮", value=cur_sub.get("karaoke", True),
                )

            gr.HTML('<div class="section-title">🤖 LLM 文案</div>')
            with gr.Row():
                llm_provider = gr.Textbox(
                    label="LLM Provider", value=cur_llm.get("provider", "agnes"),
                )
                llm_key = gr.Textbox(
                    label="LLM API Key", value=cur_llm.get("api_key", ""), type="password",
                )
                llm_model = gr.Textbox(
                    label="LLM 模型", value=cur_llm.get("model", ""),
                )

            gr.HTML('<div class="section-title">📦 发布</div>')
            with gr.Row():
                pub_mode = gr.Dropdown(
                    label="发布模式",
                    choices=[("手动", "manual"), ("半自动", "semi_auto"), ("全自动", "auto")],
                    value=cur_pub.get("mode", "semi_auto"),
                )

            with gr.Row():
                save_btn = gr.Button("💾 保存设置（热生效）", variant="primary", size="lg")
                reset_btn = gr.Button("↩️ 恢复默认", size="lg")
            settings_status = gr.Textbox(label="保存结果", interactive=False)

            def _save(t_provider, t_threads, t_builtin, w_python,
                      g_enabled, g_weight, g_stride,
                      s_font, s_size, s_margin, s_karaoke,
                      l_provider, l_key, l_model, p_mode):
                msgs = []
                r = settings_mgr.update_section("tts", {
                    "provider": t_provider,
                    "moss_nano": {
                        "cpu_threads": int(t_threads),
                        "builtin_voice": t_builtin,
                    },
                })
                msgs.append(r.get("message", ""))
                r = settings_mgr.update_section("avatar", {
                    "wav2lip": {"env_python": w_python},
                    "gfpgan": {
                        "enabled": bool(g_enabled),
                        "weight_path": g_weight,
                        "stride": int(g_stride),
                    },
                })
                msgs.append(r.get("message", ""))
                r = settings_mgr.update_section("subtitle", {
                    "font_path": s_font, "font_size": int(s_size),
                    "margin_v": int(s_margin), "karaoke": bool(s_karaoke),
                })
                msgs.append(r.get("message", ""))
                r = settings_mgr.update_section("llm", {
                    "provider": l_provider, "api_key": l_key, "model": l_model,
                })
                msgs.append(r.get("message", ""))
                r = settings_mgr.update_section("publisher", {"mode": p_mode})
                msgs.append(r.get("message", ""))
                return "✅ " + " | ".join(msgs)

            def _reset():
                try:
                    settings_mgr.reset_all()
                    return "✅ 已恢复默认设置，刷新页面查看新值"
                except Exception as e:
                    return f"❌ 恢复失败: {e}"

            save_btn.click(
                _save,
                inputs=[tts_provider, tts_threads, moss_builtin, wav2lip_python,
                        gfpgan_enabled, gfpgan_weight, gfpgan_stride,
                        sub_font, sub_size, sub_margin, sub_karaoke,
                        llm_provider, llm_key, llm_model, pub_mode],
                outputs=[settings_status],
            )
            reset_btn.click(_reset, outputs=[settings_status])

        # ============ Tab 7: 任务管理 ============
        with gr.Tab("📋 任务管理"):
            gr.Markdown("### 📋 历史任务")
            with gr.Row():
                refresh_jobs_btn = gr.Button("🔄 刷新任务列表")
                jobs_limit = gr.Slider(label="显示数量", minimum=10, maximum=200,
                                       value=50, step=10)
            jobs_table = gr.Dataframe(
                headers=["任务ID", "状态", "耗时(s)", "成片", "创建时间"],
                datatype=["str", "str", "number", "str", "str"],
                value=[], interactive=False, wrap=True,
            )
            with gr.Row():
                rerun_job_id = gr.Textbox(label="任务ID（断点续跑）", scale=2)
                rerun_btn = gr.Button("▶️ 续跑", size="sm")
                del_job_id = gr.Textbox(label="任务ID（删除）", scale=2)
                del_btn = gr.Button("🗑️ 删除", size="sm")
            job_action_out = gr.Textbox(label="操作结果", interactive=False)
            job_detail_out = gr.JSON(label="任务详情", value=None)

            def _find_job(jid):
                jobs = app.list_jobs(limit=200)
                for j in jobs:
                    if j.get("job_id", "").startswith(jid) or j.get("job_id") == jid:
                        return j.get("job_id")
                return None

            def _refresh_jobs(limit):
                jobs = app.list_jobs(limit=int(limit))
                rows = []
                for j in jobs:
                    out = j.get("output") or {}
                    rows.append([
                        j.get("job_id", "")[:12],
                        j.get("status", ""),
                        round(j.get("elapsed", 0) or 0, 1),
                        out.get("final_video", "") or "",
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(j.get("created_at", 0))),
                    ])
                return rows

            def _rerun(jid):
                full = _find_job(jid)
                if not full:
                    return f"❌ 未找到任务 {jid}", None
                ok = app.rerun_job(full)
                return f"{'✅ 续跑完成' if ok else '❌ 续跑失败'}: {full}", app.get_job(full)

            def _del(jid):
                full = _find_job(jid)
                if not full:
                    return f"❌ 未找到任务 {jid}", None
                app.delete_job(full)
                return f"✅ 已删除 {full}", None

            refresh_jobs_btn.click(_refresh_jobs, inputs=[jobs_limit], outputs=[jobs_table])
            demo.load(_refresh_jobs, inputs=[jobs_limit], outputs=[jobs_table])
            rerun_btn.click(_rerun, inputs=[rerun_job_id],
                            outputs=[job_action_out, job_detail_out])
            del_btn.click(_del, inputs=[del_job_id],
                          outputs=[job_action_out, job_detail_out])

    return demo


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
    """启动 Gradio 服务"""
    if gr is None:
        raise RuntimeError("gradio 未安装，请运行: pip install gradio")
    demo = _build_ui()
    logger = get_logger()
    logger.info(f"启动 Gradio 服务: http://{host}:{port}")
    demo.queue().launch(server_name=host, server_port=port, share=share,
                        inbrowser=True, show_error=True,
                        css=CUSTOM_CSS, theme=gr.themes.Soft())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    launch(host=args.host, port=args.port, share=args.share)
