# EnlyAI 多卡并行生成单条视频 — 设计文档

> 场景：企业内部合成视频平台，目标硬件 86 核 / 8×RTX 3090。
> 需求：**多路显卡并发生成同一条视频**（降低单条视频出片延迟）。
>
> 决策（已定）：
> - 优化目标 = **单条视频最快出片**（intra-video 时间分片，非多条并行）
> - 分片策略 = **按句子/静音边界切**（复用现有 TTS 分句 + 段级时间戳）
> - 部署拓扑 = **常驻 worker 池（每卡一进程，模型只加载一次）+ 中心调度器**

---

## 1. 核心思路：沿时间轴分片唇形推理

一条视频的耗时 **~95% 在数字人唇形推理（LatentSync 扩散）**，TTS/字幕/合成都很轻。因此"多卡加速一条视频" = **把唇形推理这一段拆到 N 张卡并行**。

唇形同步是**时间局部**的（某帧嘴型只依赖附近音频窗口，不依赖整条视频），所以可沿时间轴分片、各段独立推理、再按序拼接：

```
文案 → TTS(整条, 产出段级 timestamps) → 字幕
                    │
          ┌─────────┴─────────  scatter：按句子边界分成 K 段
          │   每段 = {音频切片, 参考视频时间窗口 [t0,t1], 段序号}
          ▼
   ┌──────────┬──────────┬──────────┬──────────┐
  worker0    worker1    worker2    worker3   ← 每卡一个常驻 LatentSync
  (GPU0)     (GPU1)     (GPU2)     (GPU3)       进程，模型仅加载一次
   └──────────┴──────────┴──────────┴──────────┘
          │  gather：按段序号收齐
          ▼
   拼接(concat, 接缝在静音处) → 整条唇形视频 → 合成(字幕+BGM+画中画) → 成片
```

- 加速比 ≈ **N×**（减分片/拼接/尾部不均衡的开销），作用在最长的瓶颈段上。
- **纯编排层改动，不碰 LatentSync 模型内部**（不做 tensor/data-parallel 扩散——那要改模型、显存翻倍、还引入跨卡通信，性价比极低且脆弱）。

---

## 2. 组件设计

### 2.1 常驻 LatentSync Worker（每卡一个进程）

替换现状 `avatar_server` 的「每请求 subprocess 冷加载 `scripts/inference.py`」（分片后冷启动会被放大 K 倍，必须先解决）。

- 启动：`CUDA_VISIBLE_DEVICES=i python -m krvoiceai.api.avatar_worker --port 80{10+i}`，
  进程内**加载一次** LatentSync（UNet + VAE + audio encoder），常驻。
- 新接口（段级）：
  ```
  POST /api/avatar/generate_segment
  { "audio_base64": <该段音频>, "avatar_id": "anchor_wang",
    "ref_start": 12.30, "ref_end": 18.75,      # 参考视频时间窗口（秒）
    "seg_index": 3, "inference_steps": 25, "resolution": 512,
    "seed": 1247 }                              # 固定 seed 保证跨段可复现
  → { "video_base64": <该段唇形视频>, "seg_index": 3, "duration": 6.45 }
  ```
  worker 内部：`ffmpeg -ss ref_start -t (ref_end-ref_start)` 切出参考视频切片 → 对
  (切片, 段音频) 跑 LatentSync → 返回段视频。**只传音频，参考视频常驻本地形象库**，payload 小。
- 保留 `/health`（`backend_ready` + 当前在途段数），供调度器做负载感知。
- 兼容：保留旧 `/api/avatar/generate`（整条）走同一常驻模型，非分片路径不受影响。

### 2.2 GPU Worker 池 + 中心调度器

在 `core/gpu_runner.py` 之上（或新增 `core/gpu_pool.py`）：

- **注册表**：从 config 读 worker 列表 `avatar_workers: [http://127.0.0.1:8010, ...8017]`（8 卡=8 条）。
- **调度**：`submit_segment(seg)` → 选**在途段数最少且 /health ok** 的 worker 分发；维护每 worker 的 in-flight 计数。
- **容错**：段失败/worker 超时 → 自动重派到另一 worker（幂等，段有序号）；worker 连续失败 N 次 → 短路摘除一段时间（熔断）。
- **连接**：模块级共享 `httpx.Client`（连接池 keepalive）+ 指数退避重试。
- 这套池化 = 评估里企业级地基那条「gpu_runner endpoint 池 + 健康 LB」，一并落地。

### 2.3 段级 Scatter-Gather（avatar 阶段）

`avatar_engine` 新增 `_generate_sharded(ctx, avatar_id, output_path)`，当 `avatar.provider=latentsync` 且 `avatar.sharding.enabled=true` 时走它替代 `_generate_cloud`：

```
segments = plan_segments(ctx.tts_timestamps, ctx.audio_duration, n_workers, cfg)  # §3
seg_audios = slice_audio(ctx.audio_path, segments)                               # ffmpeg 按边界切
futures = [pool.submit_segment(seg) for seg in segments]   # 并发提交 K 段
seg_videos = gather_in_order(futures)                      # 按 seg_index 收齐
merged = stitch(seg_videos, ctx.audio_path, output_path)   # §4 拼接 + 整条音频覆盖
```

`orchestrator` 的 avatar 步骤内部并发即可，无需改整体线性编排（scatter-gather 封装在该步骤内）。

---

## 3. 分片算法（句子/静音边界 + 负载均衡）

输入：TTS 段级 `timestamps`（每句 start/end，句间有 `pause_duration` 静音）。

规则：
1. **切点只落在句间静音**（`timestamps[i].end` 与 `timestamps[i+1].start` 之间的停顿中点）→ 接缝处嘴型闭合/中性，几乎无瑕疵。
2. **最小段时长**（`min_seg_sec`，默认 2.5s）：LatentSync 有时间窗口，太短段效率低、边界效应大 → 合并相邻短句直到达标。
3. **最大段时长 / 目标段数**：目标 `K ≈ n_workers`（或 `1.5×n_workers` 做负载均衡，缓解尾部长段拖慢整体）。在满足最小段时长前提下，尽量让各段**时长均衡**（贪心分配句子到 K 桶，均衡总时长）。
4. **尾部效应**：整体延迟 = 最长段的推理时间。所以均衡段时长比"段数最多"更重要。段数 > 卡数时多出的段排队，仍受最长段约束。

产物：`[{seg_index, t_start, t_end, ref_start, ref_end, text}]`（时间轴同参考视频与音频，一致）。

---

## 4. 参考视频切片、拼接与接缝

- **参考视频切片**：参考是真人口播视频时，每段用其**对应时间窗口**的切片 → 拼接后头动/表情天然连续。参考是静态图时，所有段共用，更简单。
- **音频权威**：**不逐段拼音频**（会累积漂移）。拼接段视频后，用 **整条 TTS 音频覆盖**音轨（复用上轮 cut 模式已实现的 `concat 视频 + -map 1:a 覆盖` 模式）→ 音画全局对齐、零漂移。
- **接缝**：因切点在静音处，直接 `concat` 通常无可见跳变；保险起见在每个接缝做 **1–2 帧交叉淡化**（cfg 可关）。
- **参数一致性**：所有 worker 用相同 `inference_steps / resolution / seed / fps`，保证各段风格与画质一致。

---

## 5. 调度与容错细节

| 关注点 | 处理 |
|--------|------|
| 段失败 | 重派到其他空闲 worker（有序号，幂等）；重试上限后整 job 失败 |
| worker 崩溃/超时 | /health 摘除 + 其未完成段回队重派；worker 恢复自动回池 |
| 长尾段拖慢 | §3 均衡段时长；可选"投机执行"——最后 1 段在两卡同时跑取先返回 |
| 确定性 | 固定 seed；同输入产出稳定，便于复现与回归 |
| 单卡 OOM | worker 端 Semaphore=1（一次只跑一段），调度器不超发 |
| 多条视频同时来 | 本目标=单条最快 → 一条视频独占全池，其余 FIFO 排队（后续可加"空闲卡分给在跑视频"的动态模式） |

---

## 6. 代码触点

| 文件 | 改动 |
|------|------|
| `api/avatar_worker.py`（新增） | 常驻 LatentSync 进程 + `/generate_segment`；由现 `avatar_server.py` 的 LatentSync 封装演进而来（去掉每请求冷加载） |
| `core/gpu_pool.py`（新增） | worker 注册表 + 负载感知调度 + 容错/熔断 + 共享 httpx |
| `modules/avatar_engine.py` | 新增 `_generate_sharded`；`avatar.sharding.enabled` 时替代 `_generate_cloud` |
| `core/audio_utils.py` / 新 `sharding.py` | `plan_segments()` 分片规划 + `slice_audio()` 按边界切音频 |
| `core/ffmpeg_utils.py` | 段视频 `concat + 交叉淡化 + 整条音频覆盖`（复用已有 concat/remux） |
| `config/default.yaml` | `avatar.sharding: {enabled, min_seg_sec, target_segments, crossfade_frames}`；`gpu_runner.avatar_workers: [...]` |
| `scripts/` + `docs/GPU_UPGRADE.md` | 8 卡常驻 worker 启动脚本 + compose 钉卡 |

> 依赖前置：本设计需要先有**可跑的 LatentSync 常驻推理**（GPU_UPGRADE.md 的 P2 从"subprocess 冷加载"升级到"常驻 worker"）。二者合并做最省事。

---

## 7. 加速模型与调参

- 理想加速 ≈ `min(K, N) × (推理时间占比)`；实测受**最长段**与固定开销约束。
- 例：一条 3 分钟视频，若单卡唇形推理 30 分钟，均衡切成 8 段 → 各段 ~3.75 分钟推理，理论 8×3090 并行 ≈ **4–5 分钟**（含切片/拼接/尾部）。
- 关键调参：`min_seg_sec`（太小→效率低+接缝多；太大→并行度不足）、`target_segments`（=卡数 or 1.5×卡数）、`inference_steps`（质量/速度权衡，已有）。
- 建议先在真机跑一次基准：单卡整条 vs 8 卡分片，记录**端到端时延 + 每卡利用率 + 接缝画质**，写入 docs。

---

## 8. 分阶段落地

- [ ] **M1 常驻 worker**：`avatar_worker.py` 加载一次模型 + `/generate_segment`（先单卡，验证段推理与参考切片正确、接缝画质）。
- [ ] **M2 池 + 调度**：`gpu_pool.py` 多 worker 负载感知分发 + 容错；`avatar_engine._generate_sharded` scatter-gather 打通端到端。
- [ ] **M3 分片规划 + 拼接**：`plan_segments` 均衡算法 + 切音频 + concat/交叉淡化/音频覆盖；接缝画质调优。
- [ ] **M4 8 卡基准 + 调优**：真机跑满 8 卡，均衡尾部，出性能基线；固定 seed 回归。
- [ ] **M5 韧性**：投机执行长尾段、熔断、worker 崩溃演练、`/metrics`（每卡利用率+段耗时）。

---

## 9. 风险与取舍

- **接缝画质**是单条分片的固有风险；靠"静音处切 + 交叉淡化 + 参考视频切片对齐"控制。若真机上仍可见，退路：增大重叠、或对 seam 段做局部重生成。**建议 M1 先用真实素材出小样确认再铺开。**
- **段时长不均衡**→尾部拖慢；靠均衡分片 + 投机执行缓解。
- **单条独占全池**在多用户企业场景下利用率可能不满；本期按"单条最快"定，后续可加动态调度（队列空时多卡加速单条，繁忙时每卡一条）——数据结构（worker 池 + 段队列）已为此预留。
- 本设计**不追求跨卡单次扩散并行**（复杂、脆弱、收益有限）；如未来单段仍嫌慢，再评估 LatentSync 自身的 batch/半精度/蒸馏加速。

---

## 附：与企业级改造的关系

本设计顺带落地了质量评估中"性能与 GPU"维度的几项 P0（常驻 worker、endpoint 池 + 健康 LB、服务端并发保护、/metrics）。企业内部生产用途下，评估中原先标为"参赛过度投资"的**鉴权、CI、回归测试、可观测性、依赖锁定**均应重新纳入必做项——建议本多卡能力与 P0 稳定性地基（CI + FFmpeg 超时 + SQLite 加固）并行推进。
