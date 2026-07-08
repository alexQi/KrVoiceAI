"""GPU Worker 池 + 负载感知调度（M2：单条视频多卡分片）

把一条视频的多个音频段并发分发到多个常驻 avatar worker（每卡一个进程），
按「在途段数最少 + 健康」选 worker，段失败自动重派到其他 worker。

worker 列表来自 config：gpu_runner.avatar_workers = [http://avatar-0:8010, ...]
未配置时回退到单端点 gpu_runner.avatar_endpoint。
"""
from __future__ import annotations

import threading
from typing import Any, Optional

import httpx

from .config import get_config
from .logger import get_logger


class GPUWorkerPool:
    """avatar worker 池：负载感知分发段级推理请求。"""

    def __init__(self, endpoints: Optional[list[str]] = None):
        cfg = get_config()
        self.logger = get_logger().bind(component="gpu_pool")
        eps = endpoints or cfg.get("gpu_runner.avatar_workers", []) or []
        if not eps:
            single = cfg.get("gpu_runner.avatar_endpoint")
            eps = [single] if single else []
        self.endpoints: list[str] = [e.rstrip("/") for e in eps]
        self._inflight: dict[str, int] = {e: 0 for e in self.endpoints}
        self._down_until: dict[str, float] = {}   # 熔断：失败副本短路到此时间戳
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))
        self.logger.info(f"GPU worker 池初始化：{len(self.endpoints)} 个 worker {self.endpoints}")

    @property
    def size(self) -> int:
        return len(self.endpoints)

    def _healthy(self, ep: str) -> bool:
        try:
            r = self._client.get(f"{ep}/health", timeout=5.0)
            return r.status_code == 200 and r.json().get("backend_ready", False)
        except Exception:
            return False

    def _pick(self, exclude: set[str], clock: float) -> Optional[str]:
        """选在途最少、未熔断、健康的 worker。"""
        with self._lock:
            cands = [
                e for e in self.endpoints
                if e not in exclude and self._down_until.get(e, 0) <= clock
            ]
            if not cands:
                return None
            cands.sort(key=lambda e: self._inflight[e])
            # 只对候选做健康检查（在途最少优先）
            for e in cands:
                if self._healthy(e):
                    self._inflight[e] += 1
                    return e
        return None

    def _release(self, ep: str, failed: bool, clock: float) -> None:
        with self._lock:
            self._inflight[ep] = max(0, self._inflight[ep] - 1)
            if failed:
                # 熔断 30s
                self._down_until[ep] = clock + 30.0

    def submit_segment(self, payload: dict[str, Any], retries: int = 2,
                       clock: float = 0.0) -> dict:
        """把一个段级请求发给最优 worker；失败重派到其他 worker。

        clock: 单调时钟（秒），由调用方传入（脚本环境不能用 time.monotonic 的话传 0 关闭熔断）。
        """
        import time as _t
        tried: set[str] = set()
        last_err = "no worker available"
        for attempt in range(retries + 1):
            now = _t.monotonic()
            ep = self._pick(tried, now)
            if ep is None:
                # 所有 worker 都在途/熔断/不健康：等一下重试
                if attempt < retries:
                    _t.sleep(2.0)
                    continue
                break
            try:
                r = self._client.post(f"{ep}/api/avatar/generate_segment", json=payload)
                r.raise_for_status()
                self._release(ep, failed=False, clock=now)
                return r.json()
            except Exception as e:
                last_err = f"{ep}: {e}"
                self.logger.warning(f"段 {payload.get('seg_index')} 在 {ep} 失败，重派：{e}")
                self._release(ep, failed=True, clock=now)
                tried.add(ep)
        raise RuntimeError(f"段 {payload.get('seg_index')} 所有 worker 均失败：{last_err}")

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
