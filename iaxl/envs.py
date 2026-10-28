# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if not v:
        return default
    v = v.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return default


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _float_or_none(name: str) -> Optional[float]:
    v = os.environ.get(name)
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _watermark(name: str, default: float) -> float:
    v = _float_or_none(name)
    if v is None:
        return default
    if 1.0 < v <= 100.0:
        v /= 100.0
    if v <= 0.0 or v > 1.0:
        return default
    return v


class Envs:
    def __init__(self) -> None:
        self.IAXL_DEBUG = _bool("IAXL_DEBUG")

        self.IAXL_KV_COMPRESSION = _bool("IAXL_KV_COMPRESSION", True)
        if self.IAXL_KV_COMPRESSION and not (
            _bool("IAXL_QAT_ZIP_ENABLE", True)
            or _bool("IAXL_IAA_ZIP_ENABLE")
            or _bool("IAXL_CPU_ZIP_ENABLE", False)
        ):
            raise ValueError(
                "IAXL_KV_COMPRESSION=1 requires at least one zip backend; "
                "enable IAXL_QAT_ZIP_ENABLE/IAXL_IAA_ZIP_ENABLE/IAXL_CPU_ZIP_ENABLE "
                "or set IAXL_KV_COMPRESSION=0"
            )
        self.IAXL_DSA_GD_ENABLE = _bool("IAXL_DSA_GD_ENABLE")
        self.IAXL_KV_DATA_SHUFFLE = _bool("IAXL_KV_DATA_SHUFFLE")
        self.IAXL_KV_LOSSY_TRUNC = max(0, _int("IAXL_KV_LOSSY_TRUNC", 0))
        self.IAXL_CACHE_DIR = _str("IAXL_CACHE_DIR", "_data/kvcache")
        self.IAXL_CACHE_STREAM_SYNC_ON_GET = _bool("IAXL_CACHE_STREAM_SYNC_ON_GET")
        self.IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS = max(
            0, _int("IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS", 0)
        )

        self.IAXL_DDR_POOL_SIZE_GB = _float_or_none("IAXL_DDR_POOL_SIZE_GB")
        self.IAXL_SCRATCH_POOL_SIZE_GB = _int("IAXL_SCRATCH_POOL_SIZE_GB", 8)

        self.IAXL_CACHE_AUTO_SPILL_ENABLE = _bool("IAXL_CACHE_AUTO_SPILL_ENABLE", False)
        self.IAXL_CACHE_AUTO_SPILL_EVICT_ONLY = _bool(
            "IAXL_CACHE_AUTO_SPILL_EVICT_ONLY", False
        )
        self.IAXL_CACHE_HIGH_WATERMARK = _watermark("IAXL_CACHE_HIGH_WATERMARK", 0.80)
        self.IAXL_CACHE_LOW_WATERMARK = _watermark("IAXL_CACHE_LOW_WATERMARK", 0.60)
        if self.IAXL_CACHE_LOW_WATERMARK >= self.IAXL_CACHE_HIGH_WATERMARK:
            self.IAXL_CACHE_LOW_WATERMARK = max(
                0.0, self.IAXL_CACHE_HIGH_WATERMARK - 0.05
            )
        self.IAXL_CACHE_SPILL_BATCH_GROUPS = max(
            1, _int("IAXL_CACHE_SPILL_BATCH_GROUPS", 16)
        )
        self.IAXL_CACHE_SPILL_PACE_MS = max(0, _int("IAXL_CACHE_SPILL_PACE_MS", 2))
        self.IAXL_CACHE_EVICT_GRACE_MS = max(0, _int("IAXL_CACHE_EVICT_GRACE_MS", 50))

        self.IAXL_PREALLOC_LIMIT = _int("IAXL_PREALLOC_LIMIT", 0)

        self.IAXL_API_TIMEOUT = _int("IAXL_API_TIMEOUT", 60)
        self.IAXL_API_CONTROLLER_PORT = _int("IAXL_API_CONTROLLER_PORT", 18700)
        self.IAXL_API_WORKER_BASE_PORT = _int("IAXL_API_WORKER_BASE_PORT", 18800)

        self.IAXL_RDMA_ENABLE = _bool("IAXL_RDMA_ENABLE")
        # Comma lists (a single IP is allowed): rank r uses entry r % len; entry 0 is also the control-plane IP.
        self.IAXL_RDMA_DAEMON_NIC_IPS = _str("IAXL_RDMA_DAEMON_NIC_IPS", "")
        self.IAXL_RDMA_CLIENT_NIC_IPS = _str("IAXL_RDMA_CLIENT_NIC_IPS", "")
        self.IAXL_RDMA_DAEMON_PORT = _int("IAXL_RDMA_DAEMON_PORT", 5555)
        self.IAXL_RDMA_TP_SIZE = _int("IAXL_RDMA_TP_SIZE", 1)

        self.IAXL_PROFILE_MODE = _str("IAXL_PROFILE_MODE", "").strip().lower()
        self.IAXL_METRICS_ENABLED = _bool("IAXL_METRICS_ENABLED")
        self.IAXL_PERFETTO_ENABLED = _bool("IAXL_PERFETTO_ENABLED")


envs = Envs()
