# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import math
import os
import threading
import time
import torch
import numpy as np
from typing import Callable, Dict, List, Mapping, Optional, Set, TYPE_CHECKING, Union, cast
import psutil
from ..envs import envs
from ..kvflow import KVFlow, Task, get_accelerator_device
from ..remote_pool.remote_tensor import RemoteTensor
from ..utils.profiler import (
    profile_scope,
    profile_cross_scope,
    profile_func,
    start_profiling,
    stop_profiling,
)
from .api import (
    ControllerServer,
    register as mgmt_register,
    start_mgmt_server,
    stop_mgmt_server,
)

logger = logging.getLogger(__name__)


def _get_default_cache_size_gb() -> float:
    available_bytes = psutil.virtual_memory().available
    return available_bytes / (10 * 1024**3)


def _abort_on_error(fn):
    # An exception would only end the evict thread, leaving ranks diverged and has() possibly stuck.
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BaseException:
            logger.critical(
                "DDR pool evict failed, ranks may diverge; aborting", exc_info=True
            )
            os.abort()

    return wrapper


class KVStoreLocal:
    LABEL = "kv"

    def __init__(
        self,
        model_name: str,
        block_dim: Optional[int] = None,
        kv_caches: Optional[Mapping[str, Union[torch.Tensor, RemoteTensor]]] = None,
        layer_names: Optional[List[str]] = None,
        rank: int = 0,
        tp_size: int = 1,
        shared_mode: bool = False,
    ):

        if kv_caches is None and layer_names is None:
            raise ValueError(
                "At least one of kv_caches or layer_names must be provided"
            )

        if kv_caches is not None and block_dim is None:
            raise ValueError("block_dim is required when kv_caches is provided")

        if kv_caches is not None and layer_names is not None:
            kv_keys = set(kv_caches.keys())
            layer_set = set(layer_names)
            if kv_keys != layer_set:
                raise ValueError(
                    f"kv_caches keys {kv_keys} must match layer_names {layer_set}"
                )

        self.kv_caches = kv_caches
        self.block_dim = block_dim
        self.rank = rank
        self.tp_size = tp_size

        if kv_caches is not None:
            self.layer_names = list(kv_caches.keys())
        else:
            self.layer_names = layer_names
        # Layer index of each layer: the sort key of its load tasks.
        self._layer_order: Dict[str, int] = {}
        for index, name in enumerate(self.layer_names):
            self._layer_order[name] = index

        self.skip_compression_count = min(
            envs.IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS, len(self.layer_names)
        )

        self.has_only_mode = kv_caches is None

        if kv_caches:
            first_tensor = next(iter(kv_caches.values()))
            self.kvcache_shape = list(first_tensor.shape)

            self.block_shape = list(first_tensor.shape)
            self.block_shape[block_dim] = 1
            self.block_shape = tuple(
                self.block_shape[i]
                for i in range(len(self.block_shape))
                if i != block_dim
            )
        else:
            self.kvcache_shape = None
            self.block_shape = None

        # Cache directory keyed by the caller-supplied rank (KVStore is DP-agnostic).
        final_persist_dir = f"{model_name}_rank{rank}"

        if self.has_only_mode:
            pool_size_gb = 0.0
        else:
            env_size = envs.IAXL_DDR_POOL_SIZE_GB
            if env_size is not None:
                pool_size_gb = env_size
            else:
                pool_size_gb = _get_default_cache_size_gb()

        self.tensorzip = KVFlow(
            persist_dir=final_persist_dir,
            cache_size_gb=pool_size_gb,
            rank=rank,
            kv_caches=kv_caches,
            block_dim=block_dim,
        )
        start_profiling()

        logger.info(
            "KVStore initialized successfully: "
            "model_name=%s, rank=%d, has_only_mode=%s, "
            "num_layers=%d, block_dim=%s, block_shape=%s, "
            "pool_size_gb=%.2f, persist_dir=%s",
            model_name,
            self.rank,
            self.has_only_mode,
            len(self.layer_names),
            self.block_dim,
            self.block_shape,
            pool_size_gb,
            final_persist_dir,
        )

        role = "controller" if self.has_only_mode else "worker"
        self._shared_mode = shared_mode
        self._shared_meta_lock = threading.Lock()
        # group_key -> [accessor_mask, access_count, last_access]
        self._shared_meta: Dict[str, list] = {}
        self._spill_handler: Optional[Callable[[int, int, int], None]] = None
        self._worker_mem_lock = threading.Lock()
        self._evict_cond = threading.Condition()  # has() vs. the scheduler's evict decision
        self._evicting: Set[str] = set()  # group keys the workers are still dropping
        self._evict_enabled = True  # auto-evict switch, POST /v1/cache/evict
        if role == "worker":
            mgmt_register("GET", "/v1/cache/status", lambda params, s=self: s.status())
            mgmt_register(
                "GET",
                "/v1/cache/simple_status",
                lambda params, s=self: s.simple_status(),
            )
            mgmt_register(
                "GET",
                "/v1/health",
                lambda params, s=self: {"status": "ok", "rank": s.rank},
            )
            mgmt_register(
                "POST",
                "/v1/cache/persist",
                lambda body, s=self: {
                    "rank": s.rank,
                    "result": s.persist(int(body.get("count", 10))),
                },
            )

            def evict_labels(body, s=self):
                labels = body["labels"]
                with s._worker_mem_lock:
                    s.tensorzip.record.remove(labels)
                    s.tensorzip.mem.evict_labels(labels)  # raises -> HTTP 500 -> scheduler aborts
                return {"rank": s.rank}

            mgmt_register("POST", "/v1/cache/evict_labels", evict_labels)
            mgmt_register(
                "POST",
                "/v1/cache/evict_candidates",
                lambda body, s=self: {
                    "rank": s.rank,
                    "labels": s.get_evict_candidates(int(body.get("count", 1024))),
                },
            )
            mgmt_register(
                "GET", "/v1/cache/metrics", lambda params, s=self: s.metrics(params)
            )
        self._mgmt_server = start_mgmt_server(
            role=role, rank=self.rank, num_workers=self.tp_size
        )
        if role == "controller":
            self._controller = cast(ControllerServer, self._mgmt_server)
            # Workers start with an empty DDR pool: every row already in the DB is disk-only.
            self._evicted_until = time.time()

            def set_auto_evict(body, s=self):
                enable = body.get("enable")
                if enable is not None:
                    if not isinstance(enable, bool):
                        raise ValueError(
                            f"enable must be true or false, got {enable!r}"
                        )
                    with s._evict_cond:
                        s._evict_enabled = enable
                        s._evict_cond.wait_for(lambda: not s._evicting)  # off: in-flight eviction done
                    logger.info("DDR pool auto evict %s", "on" if enable else "off")
                return {"auto_evict": s._evict_enabled}

            self._controller.post_routes["/v1/cache/evict"] = set_auto_evict
            self._monitor_stop = threading.Event()
            threading.Thread(
                target=self._monitor_loop, name="iaxl-monitor", daemon=True
            ).start()

    def put(
        self,
        block_indices: List[int],
        block_hashs: List[str],
        layer_names: Optional[List[str]] = None,
        description: str = "",
        label: Optional[str] = None,
        pool_keys: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Task]:
        # A non-default label is its own namespace: each block hash gets one
        # more EntryGroup and record row.

        if self.has_only_mode:
            raise RuntimeError(
                "put() not available in has-only mode (kv_caches not provided)"
            )

        if layer_names is None:
            layer_names = self.layer_names

        # skip_compression_count counts GLOBAL layers, but put() may be called with
        # a single layer / subset (layerwise async), where flow compares against the
        # per-call tensor index. Rebase onto this call's first layer so "skip the
        # first N layers" holds regardless of how puts are batched.
        base = self.layer_names.index(layer_names[0])
        local_skip = max(0, self.skip_compression_count - base)

        result = self.tensorzip.put(
            label=label or self.LABEL,
            tensor_keys=layer_names,
            chunk_indices=block_indices,
            chunk_labels=block_hashs,
            description=description,
            skip_compression_count=local_skip,
            pool_keys=pool_keys,
        )

        if label is not None or self.layer_names[-1] in layer_names:
            self.tensorzip.put_finish(label or self.LABEL, block_hashs)
            self.tensorzip.record_flush()

        return result

    def put_wait(
        self,
        put_results: Dict[str, Task],
        layer_names: Optional[List[str]] = None,
        wait: bool = True,
    ) -> bool:

        if self.has_only_mode:
            raise RuntimeError(
                "put_wait() not available in has-only mode (kv_caches not provided)"
            )

        return self.tensorzip.put_wait(
            put_results,
            tensor_dict_keys=layer_names,
            wait=wait,
        )

    def get(
        self,
        block_indices: List[int],
        block_hashs: List[str],
        layer_names: Optional[List[str]] = None,
        description: str = "",
        label: Optional[str] = None,
        priorities: Optional[Dict[str, int]] = None,
        bundle_id: int = 0,
        pool_keys: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Task]:
        # priorities: OMP-Main level per layer name (None: 0); bundle_id: for set_bundles_priority.
        if self.has_only_mode:
            raise RuntimeError(
                "get() not available in has-only mode (kv_caches not provided)"
            )

        if layer_names is None:
            layer_names = self.layer_names

        sort_keys: Dict[str, int] = {}
        for name in layer_names:
            sort_keys[name] = self._layer_order[name]

        return self.tensorzip.get(
            label=label or self.LABEL,
            tensor_keys=layer_names,
            chunk_indices=block_indices,
            chunk_labels=block_hashs,
            description=description,
            priorities=priorities,
            bundle_id=bundle_id,
            sort_keys=sort_keys,
            pool_keys=pool_keys,
        )

    def set_bundles_priority(self, bundle_ids: List[int], priority: int) -> int:
        # Moves waiting load tasks of these bundles from less urgent levels to `priority`.
        return self.tensorzip.set_bundles_priority(bundle_ids, priority)

    def sort_priority_level(self, priority: int) -> None:
        # Sorts the waiting load tasks of level `priority` by layer.
        self.tensorzip.sort_priority_level(priority)

    def get_wait(
        self,
        get_results: Dict[str, Task],
        layer_names: Optional[List[str]] = None,
        wait: bool = True,
    ) -> bool:

        if self.has_only_mode:
            raise RuntimeError(
                "get_wait() not available in has-only mode (kv_caches not provided)"
            )

        return self.tensorzip.get_wait(
            get_results,
            tensor_dict_keys=layer_names,
            wait=wait,
        )

    def has(self, block_hashs: Optional[List[str]] = None,
            label: Optional[str] = None,
            truncate: bool = True,
            client_slot: int = 0) -> List[bool]:
        if not block_hashs:
            self.tensorzip.record_flush()
            return []

        label = label or self.LABEL
        with self._evict_cond:
            if self._evicting:
                keys = {f"{label}:{h}" for h in block_hashs}  # group keys, as Record.evict returns
                self._evict_cond.wait_for(lambda: self._evicting.isdisjoint(keys))
            results = self.tensorzip.has(label=label, chunk_labels=block_hashs)
            if self.has_only_mode:
                # Every hit, including those past the truncation point that the connector re-saves.
                self.tensorzip.record.touch(label, block_hashs)
                if self._shared_mode:
                    hits = [h for h, ok in zip(block_hashs, results) if ok]
                    if hits:
                        self.record_shared_access(label, hits, client_slot)
        if not truncate:
            return results

        mask = np.array(results, dtype=np.bool_)
        idx = np.argmin(mask)
        if not mask[idx]:
            mask[idx + 1 :] = False
            results = mask.tolist()

        return results

    def record_shared_access(self, label: str, block_hashes: List[str], client_slot: int = 0) -> None:
        if not self._shared_mode or not block_hashes:
            return
        now = time.time()
        bit = 1 << (client_slot & 63)
        with self._shared_meta_lock:
            for h in block_hashes:
                gk = f"{label}:{h}"
                entry = self._shared_meta.get(gk)
                if entry is None:
                    self._shared_meta[gk] = [bit, 1, now]
                else:
                    entry[0] |= bit
                    entry[1] += 1
                    entry[2] = now

    def export_compressed(self, group_keys: List[str]) -> tuple[List[str], List[bytes]]:
        """Read already-compressed SubEntry blobs for `group_keys` (label:hash) directly
        from Mem without decompression, for Host-to-Host RDMA spill."""
        if self.has_only_mode:
            raise RuntimeError("export_compressed() not available in has-only mode")
        out_keys: List[str] = []
        out_blobs: List[bytes] = []
        mem = self.tensorzip.mem
        for gk in group_keys:
            full_keys = [f"{gk}:{ln}" for ln in self.layer_names]
            try:
                blobs = mem.get(full_keys)
            except RuntimeError:
                continue
            out_keys.extend(full_keys)
            out_blobs.extend(blobs)
        return out_keys, out_blobs

    def import_compressed(self, label: str, block_hashes: List[str],
                          keys: List[str], blobs: List[bytes],
                          client_slot: int = 0) -> None:
        """Write already-compressed blobs directly into Mem and submit Record metadata,
        skipping ScratchPool and re-compression."""
        if not self.has_only_mode and keys:
            self.tensorzip.mem.put(keys, blobs)
            self.tensorzip.put_finish(label, block_hashes)
            self.tensorzip.record_flush()
        elif self.has_only_mode and self._shared_mode:
            self.record_shared_access(label, block_hashes, client_slot)

    def stop(self):
        if self.has_only_mode:
            self._monitor_stop.set()
        stop_mgmt_server(self)
        self.tensorzip.stop()

    _EVICT_BATCH = 16384

    def _select_shared_evict_labels(self, max_count: int, min_count: int, cutoff: float) -> List[str]:
        """Select eviction candidates on a shared daemon ordered by:
        (num_accessor_nodes ASC, access_count ASC, last_access ASC).
        Includes unshared/bypassed groups present in worker Mem LRU even if absent from _shared_meta."""
        with self._shared_meta_lock:
            shared_protected = {
                gk for gk, (mask, _, _) in self._shared_meta.items() if mask.bit_count() >= 2
            }
            shared_meta_snapshot = dict(self._shared_meta)

        selected: List[str] = []
        selected_set: Set[str] = set()
        if self._controller is not None:
            fetch_n = max_count + len(shared_protected) + 512
            resp = self._controller.broadcast(
                lambda c: c.post("/v1/cache/evict_candidates", {"count": fetch_n}),
                ranks=[0],
            )["0"]
            for gk in resp.get("labels", []):
                if len(selected) >= max_count:
                    break
                if gk in shared_protected:
                    continue
                meta = shared_meta_snapshot.get(gk)
                if meta is not None and meta[2] >= cutoff and len(selected) >= min_count:
                    continue
                selected.append(gk)
                selected_set.add(gk)

        if len(selected) < min_count and shared_meta_snapshot:
            items = sorted(
                shared_meta_snapshot.items(),
                key=lambda kv: (kv[1][0].bit_count(), kv[1][1], kv[1][2]),
            )
            for gk, (_mask, _cnt, last_acc) in items:
                if len(selected) >= max_count:
                    break
                if gk in selected_set:
                    continue
                if last_acc < cutoff or len(selected) < min_count:
                    selected.append(gk)
                    selected_set.add(gk)

        if selected:
            with self._shared_meta_lock:
                for gk in selected:
                    self._shared_meta.pop(gk, None)
            self.tensorzip.record.remove(selected)
        return selected

    def evict_lru(self, max_count: int, min_count: int = 0) -> int:
        # Evicts up to max_count lease-expired groups, and at least min_count groups even if
        # their lease has not expired. Returns the number of groups evicted.
        min_count = min(min_count, max_count)
        evicted = 0
        while evicted < max_count:
            with self._evict_cond:
                self._evict_cond.wait_for(lambda: not self._evicting)  # one eviction at a time
                if not self._evict_enabled:  # switched off after _monitor_loop checked
                    break
                cutoff = time.time() - envs.IAXL_DDR_POOL_EVICT_LEASE_S
                batch_max = min(self._EVICT_BATCH, max_count - evicted)
                batch_min = max(0, min(self._EVICT_BATCH, min_count - evicted))
                if self._shared_mode and self._shared_meta:
                    labels = self._select_shared_evict_labels(batch_max, batch_min, cutoff)
                else:
                    labels, self._evicted_until = self.tensorzip.record.evict(
                        self._evicted_until,
                        cutoff,
                        batch_max,
                        batch_min,
                    )
                if not labels:
                    break
                # has() on these waits until every rank has dropped them.
                self._evicting.update(labels)
            workers = self._controller.broadcast(
                lambda c: c.post("/v1/cache/evict_labels", {"labels": labels})
            )
            failed = {r: w for r, w in workers.items() if "error" in w}
            if failed:
                raise RuntimeError(f"evict_labels failed: {failed}")
            with self._evict_cond:
                self._evicting.clear()
                self._evict_cond.notify_all()
            evicted += len(labels)
        return evicted

    _MONITOR_INTERVAL_S = 10

    @_abort_on_error
    def _monitor_loop(self) -> None:
        # Rank 0's DDR pool stands for all ranks: logs its status and drives auto eviction / spill.
        last_evict_check = time.monotonic()
        last_spill_check = time.monotonic()
        spilling = False
        # _spill_handler is attached after this thread starts, so re-evaluate every tick.
        while not self._monitor_stop.wait(
            min(self._MONITOR_INTERVAL_S, envs.IAXL_DDR_POOL_SPILL_INTERVAL_S)
            if self._spill_handler is not None
            else self._MONITOR_INTERVAL_S
        ):
            st = self._controller.broadcast(
                lambda c: c.get("/v1/cache/simple_status"), ranks=[0]
            )["0"]
            if "error" in st:
                raise RuntimeError(f"DDR pool status failed: {st}")
            self._log_pool_status(st)

            now = time.monotonic()
            cap = st["capacity_bytes"]
            cur = st["current_bytes"]
            if (
                self._spill_handler is not None
                and self._evict_enabled
                and cap
                and st["group_count"]
                and now - last_spill_check >= envs.IAXL_DDR_POOL_SPILL_INTERVAL_S
            ):
                last_spill_check = now
                usage = cur / cap
                # Hysteresis: once triggered at HIGH keep spilling until usage drops to LOW.
                if usage >= envs.IAXL_DDR_POOL_SPILL_HIGH_WATERMARK or (
                    spilling and usage > envs.IAXL_DDR_POOL_SPILL_LOW_WATERMARK
                ):
                    spilling = True
                    self._spill_handler(cur, cap, st["group_count"])
                    continue
                spilling = False

            if now - last_evict_check < envs.IAXL_DDR_POOL_EVICT_INTERVAL_S:
                continue
            last_evict_check = now
            if not self._evict_enabled or not cap:  # 0 = unlimited pool
                continue
            usage = cur / cap
            if usage >= envs.IAXL_DDR_POOL_EVICT_HIGH_WATERMARK and st["group_count"]:
                # Groups are about the same size; Mem only warns when over budget, so cur may
                # exceed cap and the byte deficit (not a ratio of group_count) is what to free.
                group_bytes = cur / st["group_count"]

                def groups_to(mark: float) -> int:
                    return max(0, math.ceil((cur - mark * cap) / group_bytes))

                max_count = groups_to(envs.IAXL_DDR_POOL_EVICT_LOW_WATERMARK)
                min_count = groups_to(envs.IAXL_DDR_POOL_EVICT_FORCE_WATERMARK)
                t0 = time.monotonic()
                evicted = self.evict_lru(max_count, min_count)
                logger.info(
                    "[evict] triggered: DDR pool usage %.2f, evicted %d LRU groups "
                    "(target %d, forced at least %d) in %.3fs",
                    usage,
                    evicted,
                    max_count,
                    min_count,
                    time.monotonic() - t0,
                )

    def _log_pool_status(self, st: dict) -> None:
        gib = 1024**3
        zip_bytes = st["current_bytes"]
        cap = st["capacity_bytes"]
        # Mem only tracks cumulative zip/unzip totals; apply that ratio to what is resident now.
        ratio = (
            st["total_zip_bytes"] / st["total_unzip_bytes"]
            if st["total_unzip_bytes"]
            else 1.0
        )
        with self._evict_cond:
            evicting = len(self._evicting)
        logger.info(
            "[monitor][rank%d] | mem pool %.2f/%.2fG (%.0f%%) %d grp | "
            "zip/raw %.2f/%.2fG (%.2f) | "
            "evict %s/%d | "
            "host mem %.1f/%.1fG",
            self.rank,
            zip_bytes / gib, cap / gib, zip_bytes / cap * 100 if cap else 0.0, st["group_count"],
            zip_bytes / gib, zip_bytes / ratio / gib if ratio else 0.0, ratio,
            "on" if self._evict_enabled else "off", st["evictions"],
            st["host_mem_used_bytes"] / gib, st["host_mem_total_bytes"] / gib,
        )

    def status(self) -> dict:
        status = self.tensorzip.status()
        status["rank"] = self.rank
        status["num_layers"] = len(self.layer_names)
        status["kvcache_shape"] = self.kvcache_shape
        return status

    def simple_status(self) -> dict:
        # Just what _monitor_loop needs; skips the O(n) unpersisted_count walk of status().
        mem = self.tensorzip.mem
        vm = psutil.virtual_memory()
        with self._worker_mem_lock:
            current_bytes, group_count = mem.current_bytes, mem.group_count
        return {
            "rank": self.rank,
            "current_bytes": current_bytes,
            "capacity_bytes": mem.capacity_bytes,
            "group_count": group_count,
            "total_zip_bytes": mem.total_zip_bytes,
            "total_unzip_bytes": mem.total_unzip_bytes,
            "evictions": mem.evictions,
            "host_mem_used_bytes": vm.used,
            "host_mem_total_bytes": vm.total,
        }

    def metrics(self, params: Optional[dict] = None) -> dict:
        from ..torch_ext import (
            metrics_set_enabled,
            metrics_reset,
            metrics_read,
        )

        params = params or {}
        if "enable" in params:
            metrics_set_enabled(
                str(params["enable"]).lower() in ("1", "true", "yes", "on")
            )
        if str(params.get("reset", "")).lower() in ("1", "true", "yes", "on"):
            metrics_reset()

        result = metrics_read()
        result["rank"] = self.rank
        return result

    def persist(self, max_count: int) -> dict:
        if self.has_only_mode:
            return {"error": "persist() not available in has-only mode (scheduler)"}
        return self.tensorzip.persist(max_count)

    def evict(self, max_count: int) -> dict:
        if self.has_only_mode:
            return {"error": "evict() not available in has-only mode (scheduler)"}
        return self.tensorzip.evict(max_count)

    def get_persist_candidates(self, max_count: int) -> List[str]:
        return self.tensorzip.get_persist_candidates(max_count)

    def get_evict_candidates(self, max_count: int) -> List[str]:
        # Keep the LRU snapshot ordered with worker-side phase-two eviction.
        with self._worker_mem_lock:
            return self.tensorzip.get_evict_candidates(max_count)

    def mem_usage(self) -> tuple[int, int]:
        with self._worker_mem_lock:
            mem = self.tensorzip.mem
            return mem.current_bytes, mem.capacity_bytes


if TYPE_CHECKING:  # the two shells share one interface; pick the local one for typing
    KVStore = KVStoreLocal
elif envs.IAXL_RDMA_ENABLE and envs.IAXL_DDR_POOL_SIZE_GB == 0:
    from ..remote_pool.kvstore_remote import KVStoreRemote

    KVStore = KVStoreRemote
elif envs.IAXL_RDMA_ENABLE:
    from .hybrid import HybridKVStore

    KVStore = HybridKVStore
else:
    KVStore = KVStoreLocal
