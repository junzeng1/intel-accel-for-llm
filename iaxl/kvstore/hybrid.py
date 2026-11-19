# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hybrid two-tier KVStore: local HeadNode DDR pool + optional remote shared
daemon/cluster over RDMA/GDR.

- Local-first reads: check local DDR first, then remote for the unmatched suffix;
  remote hits are loaded directly to GPU via GDR without backfilling local DDR.
- Local-first writes with bypass: write to local DDR by default; bypass directly
  to remote via GDR when local DDR is above the spill high watermark or the
  single request exceeds remaining local capacity / bypass ratio.
- Two-phase non-blocking background spill:
  Phase 1: read already-compressed blobs from local Mem and push via
           Host-to-Host RDMA (PUT_COMPRESSED) without blocking has()/get().
  Phase 2: evict spilled labels from local DDR across all ranks.
- Optional hot-block promotion: mirror local blocks hit >= N times to remote.
"""

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import torch

from ..envs import envs
from ..remote_pool.kvstore_remote import KVStoreRemote
from .api import register as mgmt_register
from .kvstore import KVStoreLocal

logger = logging.getLogger(__name__)


@dataclass
class _HybridTask:
    """Combines local Task and/or remote task(s) for a single layer."""

    local_task: Any = None
    remote_task: Any = None
    is_put: bool = False


class HybridKVStore:
    LABEL = "kv"

    def __init__(
        self,
        model_name: str,
        block_dim: Optional[int] = None,
        kv_caches: Optional[Mapping[str, torch.Tensor]] = None,
        layer_names: Optional[List[str]] = None,
        rank: int = 0,
        tp_size: int = 1,
    ):
        self.rank = rank
        self.tp_size = tp_size
        self.has_only_mode = kv_caches is None
        self.local = KVStoreLocal(
            model_name=model_name,
            block_dim=block_dim,
            kv_caches=kv_caches,
            layer_names=layer_names,
            rank=rank,
            tp_size=tp_size,
        )
        self.layer_names = self.local.layer_names
        self.kvcache_shape = self.local.kvcache_shape
        self.block_shape = self.local.block_shape
        self.block_dim = self.local.block_dim

        self.remote: Optional[KVStoreRemote] = None
        if envs.IAXL_RDMA_DAEMON_NIC_IPS:
            try:
                self.remote = KVStoreRemote(
                    model_name=model_name,
                    block_dim=block_dim,
                    kv_caches=dict(kv_caches) if kv_caches is not None else None,
                    layer_names=layer_names,
                    rank=rank,
                    tp_size=tp_size,
                )
            except Exception:
                logger.warning(
                    "HybridKVStore(rank=%d): failed to connect remote daemon, "
                    "falling back to local-only mode",
                    rank,
                    exc_info=True,
                )
                self.remote = None
        else:
            logger.info(
                "HybridKVStore(rank=%d): IAXL_RDMA_DAEMON_NIC_IPS not set, "
                "running in local-only mode",
                rank,
            )

        # Request-level write routing cache: description -> ("local" | "remote")
        self._put_routes: Dict[str, str] = {}
        self._local_hit_counts: Dict[str, int] = {}
        self._promoted: Set[str] = set()
        self._promote_queue: Optional[queue.Queue] = None

        if self.has_only_mode:
            if self.remote is not None:
                self.local._spill_handler = self._spill_to_remote
                if envs.IAXL_HYBRID_PROMOTE_HIT_COUNT > 0:
                    self._promote_queue = queue.Queue(maxsize=4096)
                    threading.Thread(
                        target=self._promote_loop,
                        name="iaxl-hybrid-promote",
                        daemon=True,
                    ).start()
        else:
            mgmt_register("GET", "/v1/cache/status", lambda params, s=self: s.status())
            mgmt_register(
                "POST",
                "/v1/cache/spill_candidates",
                lambda body, s=self: {
                    "rank": s.rank,
                    "labels": s.local.get_evict_candidates(int(body.get("count", 256))),
                },
            )
            mgmt_register(
                "POST",
                "/v1/cache/spill_labels",
                lambda body, s=self: s._worker_spill_labels(
                    body["labels"],
                    body.get("hashes", []),
                    body.get("label", s.LABEL),
                ),
            )

    def _worker_spill_labels(
        self, group_keys: List[str], block_hashes: List[str], label: str
    ) -> dict:
        if self.remote is None:
            return {"rank": self.rank, "spilled": 0, "error": "remote unavailable"}
        keys, blobs = self.local.export_compressed(group_keys)
        if keys or block_hashes:
            self.remote.put_compressed(label, block_hashes, keys, blobs)
        return {"rank": self.rank, "spilled": len(keys)}

    def _spill_to_remote(self, cur: int, cap: int, group_count: int) -> None:
        """Two-phase non-blocking spill from local DDR to remote shared DDR."""
        if self.remote is None or not group_count:
            return
        t_start = time.monotonic()
        t_xfer = 0.0
        low_mark = envs.IAXL_DDR_POOL_SPILL_LOW_WATERMARK
        group_bytes = cur / group_count
        target_groups = max(1, math.ceil((cur - low_mark * cap) / group_bytes))
        batch_size = min(target_groups, 128)

        # Phase 1a: ask rank 0 for oldest LRU group keys (non-blocking)
        cand_resp = self.local._controller.broadcast(
            lambda c: c.post("/v1/cache/spill_candidates", {"count": batch_size}),
            ranks=[0],
        )["0"]
        if "error" in cand_resp:
            logger.warning("[spill] spill_candidates failed: %s", cand_resp)
            return
        candidates: List[str] = cand_resp.get("labels", [])
        if not candidates:
            return

        # Group by label prefix (usually "kv")
        by_label: Dict[str, List[Tuple[str, str]]] = {}
        for gk in candidates:
            if ":" in gk:
                lab, bh = gk.split(":", 1)
                by_label.setdefault(lab, []).append((gk, bh))

        spilled_labels: List[str] = []
        for lab, pairs in by_label.items():
            gks = [p[0] for p in pairs]
            bhashes = [p[1] for p in pairs]
            # Deduplicate against remote before transferring
            try:
                remote_exists = self.remote.has(bhashes, label=lab, truncate=False)
            except Exception:
                logger.warning("[spill] remote.has failed, aborting spill pass", exc_info=True)
                return
            missing_gks = [gk for gk, ex in zip(gks, remote_exists) if not ex]
            missing_hashes = [bh for bh, ex in zip(bhashes, remote_exists) if not ex]
            if missing_gks:
                t_x = time.monotonic()
                workers = self.local._controller.broadcast(
                    lambda c, mg=missing_gks, mh=missing_hashes, lb=lab: c.post(
                        "/v1/cache/spill_labels",
                        {"labels": mg, "hashes": mh, "label": lb},
                    )
                )
                failed = {r: w for r, w in workers.items() if "error" in w}
                if failed:
                    logger.warning("[spill] worker spill_labels failed: %s", failed)
                    return
                # Also notify remote scheduler of newly spilled hashes
                try:
                    self.remote.put_compressed(lab, missing_hashes, [], [])
                except Exception:
                    logger.warning("[spill] remote scheduler notify failed", exc_info=True)
                t_xfer += time.monotonic() - t_x
            spilled_labels.extend(gks)

        if not spilled_labels:
            return

        # Phase 2: remove spilled groups from local Record + evict from local Mem
        self.local.tensorzip.record.remove(spilled_labels)
        with self.local._evict_cond:
            self.local._evict_cond.wait_for(lambda: not self.local._evicting)
            self.local._evicting.update(spilled_labels)
        try:
            workers = self.local._controller.broadcast(
                lambda c: c.post("/v1/cache/evict_labels", {"labels": spilled_labels})
            )
            failed = {r: w for r, w in workers.items() if "error" in w}
            if failed:
                raise RuntimeError(f"evict_labels failed after spill: {failed}")
        finally:
            with self.local._evict_cond:
                self.local._evicting.clear()
                self.local._evict_cond.notify_all()

        logger.info("[spill] migrated %d groups to remote DDR (usage %.2f -> target %.2f) "
                    "total %.2fs, xfer %.2fs",
                    len(spilled_labels), cur / cap, low_mark,
                    time.monotonic() - t_start, t_xfer)

    def _promote_loop(self) -> None:
        """Background thread mirroring frequently hit local blocks to remote."""
        assert self._promote_queue is not None
        while not self.local._monitor_stop.is_set():
            try:
                lab, bh = self._promote_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            batch = [(lab, bh)]
            while len(batch) < 64:
                try:
                    batch.append(self._promote_queue.get_nowait())
                except queue.Empty:
                    break
            if self.remote is None:
                continue
            by_lab: Dict[str, List[str]] = {}
            for l_item, b_item in batch:
                by_lab.setdefault(l_item, []).append(b_item)
            for l_item, bhashes in by_lab.items():
                try:
                    exists = self.remote.has(bhashes, label=l_item, truncate=False)
                    missing = [b for b, ex in zip(bhashes, exists) if not ex]
                    if not missing:
                        continue
                    gks = [f"{l_item}:{b}" for b in missing]
                    workers = self.local._controller.broadcast(
                        lambda c, mg=gks, mh=missing, lb=l_item: c.post(
                            "/v1/cache/spill_labels",
                            {"labels": mg, "hashes": mh, "label": lb},
                        )
                    )
                    if not any("error" in w for w in workers.values()):
                        self.remote.put_compressed(l_item, missing, [], [])
                except Exception:
                    logger.debug("[promote] background promote failed", exc_info=True)

    # -- scheduler / lookup ----------------------------------------------------
    def has(
        self,
        block_hashs: Optional[List[str]] = None,
        label: Optional[str] = None,
        truncate: bool = True,
    ) -> List[bool]:
        if not block_hashs:
            self.local.has(block_hashs, label=label, truncate=truncate)
            return []

        lab = label or self.LABEL
        local_res = self.local.has(block_hashs, label=lab, truncate=False)

        if self.has_only_mode and self._promote_queue is not None:
            thresh = envs.IAXL_HYBRID_PROMOTE_HIT_COUNT
            for bh, ok in zip(block_hashs, local_res):
                if ok and bh not in self._promoted:
                    cnt = self._local_hit_counts.get(bh, 0) + 1
                    self._local_hit_counts[bh] = cnt
                    if cnt >= thresh:
                        self._promoted.add(bh)
                        try:
                            self._promote_queue.put_nowait((lab, bh))
                        except queue.Full:
                            pass

        if self.remote is None:
            if not truncate:
                return local_res
            for i, ok in enumerate(local_res):
                if not ok:
                    return local_res[:i] + [False] * (len(local_res) - i)
            return local_res

        if truncate:
            first_miss = next((i for i, ok in enumerate(local_res) if not ok), len(local_res))
            if first_miss == len(local_res):
                return local_res
            try:
                remote_tail = self.remote.has(block_hashs[first_miss:], label=lab, truncate=True)
                return local_res[:first_miss] + remote_tail
            except Exception:
                logger.warning("HybridKVStore: remote.has failed, using local only", exc_info=True)
                return local_res[:first_miss] + [False] * (len(local_res) - first_miss)
        else:
            miss_pos = [i for i, ok in enumerate(local_res) if not ok]
            if not miss_pos:
                return local_res
            try:
                remote_flags = self.remote.has(
                    [block_hashs[i] for i in miss_pos], label=lab, truncate=False
                )
                merged = list(local_res)
                for i, flag in zip(miss_pos, remote_flags):
                    merged[i] = flag
                return merged
            except Exception:
                logger.warning("HybridKVStore: remote.has failed, using local only", exc_info=True)
                return local_res

    # -- worker data path ------------------------------------------------------
    def _should_bypass_to_remote(self, num_blocks: int, num_layers: int) -> bool:
        if self.remote is None:
            return False
        cur, cap = self.local.mem_usage()
        if cap <= 0:
            return False
        # Condition A: local DDR usage >= spill high watermark
        if cur / cap >= envs.IAXL_DDR_POOL_SPILL_HIGH_WATERMARK:
            return True
        # Condition B: estimated request size exceeds remaining local capacity or single-req ratio
        if self.local.kv_caches:
            first = next(iter(self.local.kv_caches.values()))
            block_bytes = 1
            for d, s in enumerate(first.shape):
                if d != self.block_dim:
                    block_bytes *= s
            block_bytes *= first.element_size()
            est_bytes = num_blocks * block_bytes * len(self.layer_names)
            if est_bytes > max(0, cap - cur):
                return True
            if est_bytes > cap * envs.IAXL_HYBRID_BYPASS_SINGLE_REQ_RATIO:
                return True
        return False

    def put(
        self,
        block_indices: List[int],
        block_hashs: List[str],
        layer_names: Optional[List[str]] = None,
        description: str = "",
        label: Optional[str] = None,
    ) -> Dict[str, _HybridTask]:
        names = layer_names or self.layer_names
        route_key = f"{label or self.LABEL}:{description}"
        route = self._put_routes.get(route_key)
        if route is None:
            route = "remote" if self._should_bypass_to_remote(len(block_indices), len(names)) else "local"
            self._put_routes[route_key] = route

        if route == "remote" and self.remote is not None:
            rtasks = self.remote.put(block_indices, block_hashs, layer_names=names,
                                     description=description, label=label)
            res = {k: _HybridTask(remote_task=t, is_put=True) for k, t in rtasks.items()}
        else:
            ltasks = self.local.put(block_indices, block_hashs, layer_names=names,
                                    description=description, label=label)
            res = {k: _HybridTask(local_task=t, is_put=True) for k, t in ltasks.items()}

        if label is not None or self.layer_names[-1] in names:
            self._put_routes.pop(route_key, None)
        return res

    def put_wait(
        self,
        put_results: Dict[str, _HybridTask],
        layer_names: Optional[List[str]] = None,
        wait: bool = True,
    ) -> bool:
        keys = layer_names or list(put_results.keys())
        local_map = {k: put_results[k].local_task for k in keys if put_results[k].local_task is not None}
        remote_map = {k: put_results[k].remote_task for k in keys if put_results[k].remote_task is not None}
        ok = True
        if local_map:
            ok = self.local.put_wait(local_map, list(local_map.keys()), wait=wait) and ok
        if remote_map and self.remote is not None:
            ok = self.remote.put_wait(remote_map, list(remote_map.keys()), wait=wait) and ok
        return ok

    def get(
        self,
        block_indices: List[int],
        block_hashs: List[str],
        layer_names: Optional[List[str]] = None,
        description: str = "",
        label: Optional[str] = None,
        priorities: Optional[Dict[str, int]] = None,
        bundle_id: int = 0,
    ) -> Dict[str, _HybridTask]:
        names = layer_names or self.layer_names
        if self.remote is None:
            ltasks = self.local.get(block_indices, block_hashs, layer_names=names,
                                    description=description, label=label,
                                    priorities=priorities, bundle_id=bundle_id)
            return {k: _HybridTask(local_task=t) for k, t in ltasks.items()}

        lab = label or self.LABEL
        ltasks: Dict[str, Any] = {}
        rtasks: Dict[str, Any] = {}
        with self.local._worker_mem_lock:
            in_local = self.local.tensorzip.has(label=lab, chunk_labels=block_hashs)
            local_pos = [i for i, ok in enumerate(in_local) if ok]
            remote_pos = [i for i, ok in enumerate(in_local) if not ok]

            if local_pos:
                ltasks = self.local.get(
                    [block_indices[i] for i in local_pos],
                    [block_hashs[i] for i in local_pos],
                    layer_names=names,
                    description=description,
                    label=label,
                    priorities=priorities,
                    bundle_id=bundle_id,
                )
        if remote_pos:
            rtasks = self.remote.get(
                [block_indices[i] for i in remote_pos],
                [block_hashs[i] for i in remote_pos],
                layer_names=names,
                description=description,
                label=label,
                priorities=priorities,
                bundle_id=bundle_id,
            )
        return {
            n: _HybridTask(local_task=ltasks.get(n), remote_task=rtasks.get(n))
            for n in names
        }

    def get_wait(
        self,
        get_results: Dict[str, _HybridTask],
        layer_names: Optional[List[str]] = None,
        wait: bool = True,
    ) -> bool:
        keys = layer_names or list(get_results.keys())
        local_map = {k: get_results[k].local_task for k in keys if get_results[k].local_task is not None}
        remote_map = {k: get_results[k].remote_task for k in keys if get_results[k].remote_task is not None}
        ok = True
        if local_map:
            ok = self.local.get_wait(local_map, list(local_map.keys()), wait=wait) and ok
        if remote_map and self.remote is not None:
            ok = self.remote.get_wait(remote_map, list(remote_map.keys()), wait=wait) and ok
        return ok

    def set_bundles_priority(self, bundle_ids: List[int], priority: int) -> int:
        count = self.local.set_bundles_priority(bundle_ids, priority)
        if self.remote is not None:
            count += self.remote.set_bundles_priority(bundle_ids, priority)
        return count

    def sort_priority_level(self, priority: int) -> None:
        self.local.sort_priority_level(priority)
        if self.remote is not None:
            self.remote.sort_priority_level(priority)

    def stop(self) -> None:
        if self.remote is not None:
            self.remote.stop()
        self.local.stop()

    def status(self) -> dict:
        st = self.local.status()
        st["hybrid"] = {"remote_connected": self.remote is not None}
        if self.remote is not None:
            try:
                st["remote"] = self.remote.status()
            except Exception as e:
                st["remote_error"] = str(e)
        return st

    def metrics(self, params: Optional[dict] = None) -> dict:
        return self.local.metrics(params)

    def persist(self, max_count: int) -> dict:
        return self.local.persist(max_count)

    def evict(self, max_count: int) -> dict:
        return self.local.evict(max_count)

    def get_persist_candidates(self, max_count: int) -> List[str]:
        return self.local.get_persist_candidates(max_count)

    def get_evict_candidates(self, max_count: int) -> List[str]:
        return self.local.get_evict_candidates(max_count)
