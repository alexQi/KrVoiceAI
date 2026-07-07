# EnlyAI GPU 化 & 模型升级方案（8×RTX 3090）

> 目标机器：86 核 CPU / 8×RTX 3090（24GB×8 = 192GB 显存）
> 决策：数字人 → **LatentSync 1.5**；TTS → **CosyVoice2**；文案 LLM → 保留 DeepSeek 云 API
> 本文档 = 完整改造方案；随附 `config/user_config.yaml` 已落地「纯配置就能 GPU 化」的部分。

---

## 0. 核心架构认知（先看这个）

项目**本来就是「客户端 + GPU 推理服务」双层架构**，不是单进程跑模型：

```
┌────────────────────────────────────────┐        ┌──────────────────────────────┐
│ EnlyAI 主程序（客户端，可在任意机器）  │  HTTP  │ 8×3090 GPU 推理服务器         │
│  ├─ Web UI / Pipeline Orchestrator      │───────▶│  ├─ TTS Server  :9880         │
│  ├─ tts_engine  → gpu_runner.call_tts   │        │  │   backend=cosyvoice        │
│  ├─ avatar_engine → gpu_runner.call_avatar       │  └─ Avatar Server :8010       │
│  ├─ 字幕/查重/合成/发布（本地 CPU/GPU） │        │      backend=latentsync       │
└────────────────────────────────────────┘        └──────────────────────────────┘
```

- 客户端调用链**已经写好**：`gpu_runner.py` → `call_tts` / `call_avatar` → 打远程 HTTP。
- 服务端骨架**已存在**：`krvoiceai/api/tts_server.py`、`krvoiceai/api/avatar_server.py`，含后端切换、`/health`、`/api/*/generate`。
- 缺口只有两处：① `avatar_server._load_latentsync()` 要接 LatentSync 真实推理；② `tts_server` 要加 CosyVoice2 后端。
- 主程序和 GPU 服务器**可以是同一台**（8×3090），此时 `api_base` 填 `http://127.0.0.1:PORT` 即可，零网络开销。

因此「把 CPU 换 GPU」在本项目里 = **把重模型跑成 GPU 服务，主程序当客户端**，而不是在主进程里塞 torch。

---

## 1. 全部 CPU 环节盘点 & 改造分类

| # | 环节 | 代码位置 | 现状 | 改造方式 | 分类 |
|---|------|---------|------|---------|------|
| 1 | **数字人唇形** | `modules/avatar_engine.py` + `api/avatar_server.py` | Wav2Lip 96px，独立 CPU venv | 换 **LatentSync 1.5** GPU 服务；`provider: latentsync` + `api_base` | **换引擎（服务化）** |
| 2 | **TTS 声音克隆** | `modules/tts_engine.py` + `api/tts_server.py` | MOSS-Nano 100M ONNX，`execution_provider: cpu` | 换 **CosyVoice2** GPU 服务；新增 `cosyvoice` provider | **换引擎（服务化）** |
| 3 | **字幕 / ASR** | `modules/subtitle_engine.py:164`、`modules/script_extractor.py:307` | faster-whisper `small`+`cpu`+`int8` | `device: cuda` + `compute_type: float16` + `model_size: large-v3` | **纯配置**（本次已改） |
| 4 | **人脸增强** | `modules/avatar_engine.py:495` | GFPGAN `device: cpu`，逐帧≈9s | `device: cuda`（提速 20–50×）；配 LatentSync 后通常可不开 | **纯配置**（本次已改） |
| 5 | **视频编码** | `core/hardware_probe.py:189`、`core/ffmpeg_utils.py:40` | 已自动探测 NVENC>QSV>AMF>libx264 | **已经 GPU 优先**，无需改（服务器装带 NVENC 的 ffmpeg 即可） | ✅ 已完成 |
| 6 | **文案 LLM** | `core/llm_client.py` | DeepSeek 云 API | **保留**（按决策不自托管） | 不动 |
| 7 | 查重 SimHash / jieba | `core/text_similarity.py`、`modules/originality_checker.py` | CPU | 计算量极小，**非瓶颈**，不上 GPU | 不动 |
| 8 | 微动作 / 字幕渲染 | `modules/avatar_engine.py:647`、`ffmpeg_utils.py:1031` | 纯 FFmpeg 表达式，CPU | 轻量，保留 CPU（86 核足够） | 不动 |

**一句话**：#3#4#5 纯配置即 GPU 化（本次已落地）；#1#2 是质量主战场，需按下面 §2/§3 部署服务 + 补代码。

---

## 2. 数字人：LatentSync 1.5 落地

### 2.1 客户端（已就绪，无需改）
`avatar_engine._generate_cloud()`（`avatar_engine.py:571`）已实现完整契约：

- **请求** `POST {api_base}/api/avatar/generate`
  ```json
  {"audio_base64": "...", "avatar_id": "anchor_wang",
   "output_fps": 25, "output_resolution": [1080,1920],
   "inference_steps": 25, "resolution": 512, "config_name": "high_quality"}
  ```
- **响应** `{"video_base64": "..."}`（或 `video_url` / `data.video_base64`）+ 可选 `backend`
- 形象注册：`POST /api/avatar/register` `{avatar_id, reference_video_base64}`

`config/user_config.yaml` 里设 `avatar.provider: latentsync` + `gpu_runner.avatar_endpoint` 即接通。

### 2.2 服务端（需补代码）
`api/avatar_server.py` 已有路由 `/api/avatar/generate`、`GenerateRequest`、后端切换和 `_load_latentsync()` 骨架。**缺口**：`_load_latentsync()`（line 81）当前是 `from latentsync import LatentSyncPipeline` 的占位猜测，LatentSync 实际不是 pip 包，而是 GitHub 仓库（config + checkpoint + `scripts/inference.py` / `LipsyncPipeline`）。

**待办**：
1. `git clone https://github.com/bytedance/LatentSync`，按官方装依赖（diffusers/xformers 等），下权重。
2. 把 `_load_latentsync()` 接到真实 `LipsyncPipeline.from_pretrained(...)`，`generate(audio_path, reference_video, output_path, inference_steps, resolution)` 内部调官方推理。
3. `_get_reference_video(avatar_id)`（line 141）从形象库 `config/avatars/<id>/` 找参考视频。
4. 启动：`AVATAR_BACKEND=latentsync CUDA_VISIBLE_DEVICES=0 python -m krvoiceai.api.avatar_server --port 8010`

### 2.3 模型与显存
| 权重 | 大小 | 显存（推理） |
|------|------|-------------|
| LatentSync 1.5 UNet + audio encoder | ~5 GB | 单实例约 **8–16 GB**（`resolution=512`, `inference_steps=25`） |

单张 3090 24GB 跑 LatentSync 绰绰有余。质量对比：Wav2Lip 96px → LatentSync 512px 扩散，口型/清晰度是代际差距。

---

## 3. TTS：CosyVoice2 落地

### 3.1 客户端（需加一个 provider）
`tts_engine.py` 现有 provider：`moss_nano / mimo / gpt_sovits / edge_tts / mock`。`gpt_sovits` 已经是「打远程 TTS 服务」的 HTTP 客户端（走 `gpu_runner.call_tts` → `/api/tts/synthesize`），**CosyVoice2 复用同一套 HTTP 契约即可**。

**待办**（最小改动）：在 `tts_engine.py` 增加 `cosyvoice` 分支，复用 `gpt_sovits` 的 `call_tts` / `call_tts_register` 逻辑（二者请求体一致：`{text, voice_id, speed}` / `{voice_id, sample_audio_base64}`）。或直接把 `gpt_sovits` provider 指向 CosyVoice 服务（服务端接口统一，客户端语义无差）。

### 3.2 服务端（需补代码）
`api/tts_server.py`（152 行）目前只有 GPT-SoVITS 后端。**待办**：
1. `git clone https://github.com/FunAudioLLM/CosyVoice`，装依赖，下 `CosyVoice2-0.5B` 权重。
2. 加 `_load_cosyvoice()`：`CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, fp16=True)`。
3. `/api/tts/synthesize`：用 `inference_zero_shot(text, prompt_text, prompt_wav)` 做零样本克隆；`voice_id` → 映射到已注册的 prompt 音频。
4. `/api/tts/register_voice`：保存用户上传的 5–30s 样本作为 prompt 音频。
5. 启动：`TTS_BACKEND=cosyvoice CUDA_VISIBLE_DEVICES=1 python -m krvoiceai.api.tts_server --port 9880`

### 3.3 模型与显存
| 权重 | 大小 | 显存 |
|------|------|------|
| CosyVoice2-0.5B | ~2 GB | 单实例约 **4–6 GB**，fp16 |

克隆质量与自然度显著优于 MOSS-Nano 100M，中文强。

---

## 4. 8×3090 并行拓扑（吞吐 ×8）

当前 `pipeline.concurrency: 1`，单任务串行；`parallel_runner.py` 有骨架但未接多卡。推荐拓扑：

**方案 A（推荐，简单）——多副本 + 显卡钉绑**
- Avatar Server 起 6 个副本，各 `CUDA_VISIBLE_DEVICES=0..5`，端口 8010–8015；
- TTS Server 起 2 个副本，`CUDA_VISIBLE_DEVICES=6,7`，端口 9880–9881；
- 前面挂 nginx / 简单轮询，`api_base` 指向 LB；
- 主程序 `pipeline.concurrency: 6`（受 avatar 副本数约束），多任务并发提交。
- 效果：6 条视频同时产出，吞吐≈×6。

**方案 B（单条长视频提速）**
- 把一条视频按句/段切片，avatar 推理分发到多卡并行，再拼接。改动大，先不做。

> 注意：`concurrency` 提升后要确认 orchestrator/JobStore（SQLite）并发写安全；建议先 A 方案跑通再压并发。

---

## 5. 模型总清单与磁盘/显存预算

| 模块 | 模型 | 磁盘 | 单实例显存 | 副本数 | 卡分配 |
|------|------|------|-----------|--------|--------|
| 数字人 | LatentSync 1.5 | ~5 GB | 8–16 GB | 6 | GPU 0–5 |
| TTS | CosyVoice2-0.5B | ~2 GB | 4–6 GB | 2 | GPU 6–7 |
| 字幕/ASR | faster-whisper large-v3 | ~3 GB | ~3–5 GB（fp16） | 随主程序 | 任意共享卡 |
| 人脸增强(可选) | GFPGAN v1.4 / CodeFormer | ~0.3 GB | ~2 GB | 按需 | 共享 |
| **合计磁盘** | | **~10 GB** | | | 192GB 显存绰绰有余 |

文案 LLM = DeepSeek 云 API，不占本地显存。

---

## 6. 本次已落地（配置）

见 `config/user_config.yaml`（新建）。已把「纯配置即 GPU 化」的项切到 GPU：
- `asr.whisper`：`device: cuda` / `compute_type: float16` / `model_size: large-v3`
- `avatar.gfpgan.device: cuda`
- `avatar.wav2lip.device: cuda`（作为 LatentSync 未就绪时的 GPU 降级）
- `tts.moss_nano.execution_provider: cuda`（同上，降级用）
- 并留好 `avatar.provider: latentsync` / `tts` 服务地址的**注释开关**，待 §2/§3 服务部署后取消注释启用。

> ⚠️ 该文件是**部署目标机（8×3090）专用**。在无 GPU 机器上加载会让 whisper/gfpgan 尝试用 CUDA——请勿把它同步到无卡环境（`default.yaml` 仍是安全基线）。

---

## 7. 分阶段落地清单

- [x] **P1 配置 GPU 化**：whisper/gfpgan/moss/wav2lip device 切 GPU + NVENC（已自动）+ user_config.yaml + 本方案文档
- [x] **P2 LatentSync 代码**：`avatar_server._load_latentsync()` 接官方 `scripts/inference.py`（子进程）已补 → **部署待办**：目标机 clone + 下权重 + 起 6 副本 + 切 `provider: latentsync`
- [x] **P3 CosyVoice2 代码**：`tts_server` 加 CosyVoice2 后端（zero_shot/cross_lingual）+ `tts_engine` 加 `cosyvoice` provider 已补 → **部署待办**：clone + 下权重 + 起 2 副本 + 切 provider
- [x] **安装脚本**：`setup_cloud_gpu.sh` 补 CosyVoice2 安装块 + env（TTS_BACKEND/COSYVOICE_DIR/LATENTSYNC_DIR）
- [ ] **P4 8 卡并行**：多副本 + LB + `concurrency` 提升 + SQLite 并发验证
- [ ] **P5 可选**：whisper large-v3 → 独立 ASR 服务 / GFPGAN→CodeFormer / 自托管 Qwen（若日后要离线 LLM）

> 代码改动清单（本次）：`api/avatar_server.py`（LatentSync 子进程适配）、`api/tts_server.py`（CosyVoice2 后端，整份重写）、`modules/tts_engine.py`（cosyvoice provider 别名，复用 gpt_sovits HTTP 契约）、`scripts/setup_cloud_gpu.sh`、`config/user_config.gpu.example.yaml`（模板，git 可携带）。均通过 `py_compile` / `bash -n`。GPU 推理需在目标机实测。

---

## 8. 部署命令速查（目标机 8×3090）

```bash
# 系统依赖（含带 NVENC 的 ffmpeg）
bash scripts/setup_cloud_gpu.sh        # 已覆盖 torch(cuda12.1)+GPT-SoVITS+LatentSync+MuseTalk

# 起 6 个 LatentSync avatar 副本（GPU 0-5）
for i in 0 1 2 3 4 5; do
  AVATAR_BACKEND=latentsync CUDA_VISIBLE_DEVICES=$i \
    python -m krvoiceai.api.avatar_server --port $((8010+i)) &
done

# 起 2 个 CosyVoice2 TTS 副本（GPU 6-7）
for i in 6 7; do
  TTS_BACKEND=cosyvoice CUDA_VISIBLE_DEVICES=$i \
    python -m krvoiceai.api.tts_server --port $((9880+i-6)) &
done

# 主程序（同机 localhost 指向 LB / 单副本）
python -m krvoiceai.web.server --port 8000
```
