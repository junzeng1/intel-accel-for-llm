# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Client-side KVStore shell: same interface as `iaxl.kvstore.KVStoreLocal`, all
work forwarded to the remote_pool daemon(s) over RPC. The daemon moves KV blocks
itself (RDMA READ for put, RDMA WRITE for get); this side only registers the
kv_caches and tracks per-job completion pushed back by the daemon.

Supports:
- Multiple HeadNodes connecting to the same daemon(s) via unique client_id agent names.
- Multiple remote daemon nodes ('|' separated in IAXL_RDMA_DAEMON_NIC_IPS) via
  super-block consistent hashing.
- Host-to-Host compressed KV spill via `put_compressed`.
"""

import bisect
import hashlib
import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from ..envs import envs
from . import rpc
from .nixl_impl import parse_daemon_nodes, parse_nic_ips, rdma_xfer
from .rpc import RpcChannel, rank_port

logger = logging.getLogger(__name__)


def _client_id() -> str:
    if envs.IAXL_CLIENT_ID:
        return envs.IAXL_CLIENT_ID
    return f"{socket.gethostname()}_{os.getpid()}"


class ConsistentHashRing:
    """Deterministic consistent hash ring for routing blocks to daemon nodes."""

    def __init__(self, num_nodes: int, vnodes: int = 128):
        self.num_nodes = num_nodes
        self._ring: List[Tuple[int, int]] = []
        if num_nodes > 1:
            for idx in range(num_nodes):
                for v in range(vnodes):
                    h = int.from_bytes(
                        hashlib.blake2b(f"node{idx}#{v}".encode(), digest_size=8).digest(),
                        "little",
                    )
                    self._ring.append((h, idx))
            self._ring.sort(key=lambda x: x[0])
            self._keys = [x[0] for x in self._ring]

    def node_for(self, key: str) -> int:
        if self.num_nodes <= 1:
            return 0
        h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
        pos = bisect.bisect_left(self._keys, h)
        if pos == len(self._ring):
            pos = 0
        return self._ring[pos][1]


@dataclass
class RemoteTask:
    job_id: int
    tensor_key: str
    done: bool = False


@dataclass
class MultiRemoteTask:
    """Aggregates per-daemon RemoteTask shards for a single layer."""

    shards: List[Tuple["_RemoteNode", RemoteTask]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return all(t.done for _, t in self.shards)


class _RemoteNode:
    """Connection to one remote daemon node (scheduler or one TP rank)."""

    def __init__(
        self,
        node_idx: int,
        ip: str,
        port: int,
        client_ips: list[str],
        client_id: str,
        model_name: str,
        block_dim: Optional[int],
        kv_caches: Optional[Dict[str, torch.Tensor]],
        layer_names: Optional[List[str]],
        rank: int,
        tp_size: int,
        has_only_mode: bool,
    ):
        self.node_idx = node_idx
        self.ip = ip
        self.has_only_mode = has_only_mode
        self._pending: Dict[int, list] = {}
        self._early_done: Dict[int, set] = {}

        if has_only_mode:
            name = f"client_{client_id}_n{node_idx}_sched"
            self.peer = "daemon_sched"
            connect_port = port
        else:
            name = f"client_{client_id}_n{node_idx}_r{rank}"
            self.peer = f"daemon{rank}"
            connect_port = rank_port(port, rank)

        self.xfer = rdma_xfer(name, client_ips, rank=None if has_only_mode else rank)
        self.rpc = RpcChannel(self.xfer, self.peer, self._on_done)
        if kv_caches is not None:
            registered = set()
            for t in kv_caches.values():
                if t.data_ptr() not in registered:
                    registered.add(t.data_ptr())
                    self.xfer.register_memory(t)
        self.xfer.connect(self.peer, ip, connect_port, timeout_s=envs.IAXL_API_TIMEOUT)
        self.rpc.handshake()

        if kv_caches is not None:
            layers = {
                n: {"base": t.data_ptr(), "shape": list(t.shape),
                    "dtype": str(t.dtype).split(".")[-1], "dev_id": t.get_device()}
                for n, t in kv_caches.items()
            }
            self.rpc.call(rpc.REGISTER_KV_CACHES, rpc.pack_json({
                "layers": layers, "block_dim": block_dim, "model_name": model_name,
                "rank": rank, "tp_size": tp_size}))
            self.layer_names = list(kv_caches.keys())
        else:
            self.rpc.call(rpc.REGISTER_LAYERS, rpc.pack_json({
                "layer_names": layer_names, "model_name": model_name, "tp_size": tp_size}))
            self.layer_names = list(layer_names)
        self.layer_idx = {n: i for i, n in enumerate(self.layer_names)}

        logger.info("KVStoreRemote connected: node=%d peer=%s@%s:%d model=%s rank=%d has_only=%s",
                    node_idx, self.peer, ip, connect_port, model_name, rank, has_only_mode)

    def _on_done(self, job_id: int, layer_idx: int):
        entry = self._pending.get(job_id)
        if entry is None:
            self._early_done.setdefault(job_id, set()).add(layer_idx)
            return
        self._mark_done(job_id, layer_idx)

    def _mark_done(self, job_id: int, layer_idx: int):
        entry = self._pending[job_id]
        task = entry[0][self.layer_names[layer_idx]]
        if task.done:
            return
        task.done = True
        entry[1] -= 1
        if entry[1] == 0:
            del self._pending[job_id]

    def xfer_call(self, method, block_indices, block_hashs, names, description, label=None,
                  layer_priorities=None, bundle_id=0) -> Dict[str, RemoteTask]:
        payload = rpc.pack_blocks(block_indices, block_hashs, [self.layer_idx[n] for n in names],
                                  description, label or "", layer_priorities, bundle_id)
        job_id = int.from_bytes(self.rpc.call(method, payload), "little")
        tasks = {n: RemoteTask(job_id, n) for n in names}
        self._pending[job_id] = [tasks, len(tasks)]
        for layer_idx in self._early_done.pop(job_id, set()):
            self._mark_done(job_id, layer_idx)
        return tasks


class KVStoreRemote:
    LABEL = "kv"

    def __init__(
        self,
        model_name: str,
        block_dim: Optional[int] = None,
        kv_caches: Optional[Dict[str, torch.Tensor]] = None,
        layer_names: Optional[List[str]] = None,
        rank: int = 0,
        tp_size: int = 1,
        daemon_ip: Optional[str] = None,
        daemon_port: Optional[int] = None,
    ):
        if kv_caches is None and layer_names is None:
            raise ValueError("At least one of kv_caches or layer_names must be provided")
        if kv_caches is not None and block_dim is None:
            raise ValueError("block_dim is required when kv_caches is provided")

        default_port = daemon_port or envs.IAXL_RDMA_DAEMON_PORT
        if daemon_ip:
            daemon_nodes = [(daemon_ip, default_port, [daemon_ip])]
        else:
            daemon_nodes = parse_daemon_nodes(envs.IAXL_RDMA_DAEMON_NIC_IPS, default_port)
        client_ips = parse_nic_ips(envs.IAXL_RDMA_CLIENT_NIC_IPS, "IAXL_RDMA_CLIENT_NIC_IPS")

        self.kv_caches = kv_caches
        self.block_dim = block_dim
        self.rank = rank
        self.tp_size = tp_size
        self.has_only_mode = kv_caches is None
        self.client_id = _client_id()
        self._ring = ConsistentHashRing(len(daemon_nodes))

        self._nodes: List[_RemoteNode] = [
            _RemoteNode(
                node_idx=idx,
                ip=ctrl_ip,
                port=port,
                client_ips=client_ips,
                client_id=self.client_id,
                model_name=model_name,
                block_dim=block_dim,
                kv_caches=kv_caches,
                layer_names=layer_names,
                rank=rank,
                tp_size=tp_size,
                has_only_mode=self.has_only_mode,
            )
            for idx, (ctrl_ip, port, _) in enumerate(daemon_nodes)
        ]

        if kv_caches is not None:
            self.layer_names = list(kv_caches.keys())
            first = next(iter(kv_caches.values()))
            self.kvcache_shape = list(first.shape)
            self.block_shape = tuple(s for i, s in enumerate(first.shape) if i != block_dim)
            self._sync = (torch.xpu.synchronize if first.is_xpu
                          else lambda: torch.cuda.current_stream().synchronize())
        else:
            self.layer_names = list(layer_names)
            self.kvcache_shape = None
            self.block_shape = None
        self.layer_idx = {n: i for i, n in enumerate(self.layer_names)}

    def _route_hashes(self, block_hashes: List[str]) -> Dict[int, List[int]]:
        """Group block positions by the daemon selected from each block hash."""
        if len(self._nodes) == 1:
            return {0: list(range(len(block_hashes)))} if block_hashes else {}
        groups: Dict[int, List[int]] = {}
        for pos, block_hash in enumerate(block_hashes):
            node_idx = self._ring.node_for(block_hash)
            groups.setdefault(node_idx, []).append(pos)
        return groups

    # -- data path -------------------------------------------------------------
    def _xfer(self, method, block_indices, block_hashs, layer_names, description, what, label=None,
              priorities=None, bundle_id=0):
        if self.has_only_mode:
            raise RuntimeError(f"{what}() not available in has-only mode (kv_caches not provided)")
        names = layer_names or self.layer_names
        layer_priorities = None
        if priorities is not None:
            layer_priorities = []
            for name in names:
                if name not in priorities:
                    raise ValueError(f"priorities is missing layer {name}")
                layer_priorities.append(priorities[name])

        if len(self._nodes) == 1:
            return self._nodes[0].xfer_call(
                method, block_indices, block_hashs, names, description,
                label=label, layer_priorities=layer_priorities, bundle_id=bundle_id,
            )

        routed = self._route_hashes(block_hashs)
        agg: Dict[str, MultiRemoteTask] = {n: MultiRemoteTask() for n in names}
        for node_idx, positions in routed.items():
            node = self._nodes[node_idx]
            sub_indices = [block_indices[p] for p in positions]
            sub_hashes = [block_hashs[p] for p in positions]
            sub_tasks = node.xfer_call(
                method, sub_indices, sub_hashes, names, description,
                label=label, layer_priorities=layer_priorities, bundle_id=bundle_id,
            )
            for n, t in sub_tasks.items():
                agg[n].shards.append((node, t))
        return agg

    def put(self, block_indices, block_hashs, layer_names=None, description="",
            label=None) -> Dict[str, object]:
        if not self.has_only_mode:
            self._sync()  # attention kernels must have written kv_caches before the daemon READs
        return self._xfer(rpc.PUT, block_indices, block_hashs, layer_names, description, "put", label=label)

    def get(self, block_indices, block_hashs, layer_names=None, description="",
            label=None, priorities=None, bundle_id=0) -> Dict[str, object]:
        return self._xfer(rpc.GET, block_indices, block_hashs, layer_names, description, "get",
                          label=label, priorities=priorities, bundle_id=bundle_id)

    def put_compressed(self, label: str, block_hashes: List[str],
                       keys: List[str], blobs: List[bytes]) -> None:
        """Send already-compressed KV blobs to remote daemon(s) for Host-to-Host spill."""
        if not block_hashes:
            return
        max_payload = rpc.CTRL_BYTES // 2

        if len(self._nodes) == 1:
            self._send_compressed_node(self._nodes[0], label, block_hashes, keys, blobs, max_payload)
            return

        hash_to_node = {h: self._ring.node_for(h) for h in block_hashes}
        node_hashes: Dict[int, List[str]] = {}
        for h in block_hashes:
            node_hashes.setdefault(hash_to_node[h], []).append(h)

        node_items: Dict[int, Tuple[List[str], List[bytes]]] = {n: ([], []) for n in node_hashes}
        for k, b in zip(keys, blobs):
            parts = k.split(":", 2)
            bh = parts[1] if len(parts) >= 2 else ""
            nidx = hash_to_node.get(bh, 0)
            nk, nb = node_items.setdefault(nidx, ([], []))
            nk.append(k)
            nb.append(b)

        for nidx, bhashes in node_hashes.items():
            nk, nb = node_items[nidx]
            self._send_compressed_node(self._nodes[nidx], label, bhashes, nk, nb, max_payload)

    @staticmethod
    def _send_compressed_node(node: _RemoteNode, label: str, block_hashes: List[str],
                              keys: List[str], blobs: List[bytes], max_payload: int) -> None:
        if not keys:
            payload = rpc.pack_compressed(label, block_hashes, [], [])
            node.rpc.call(rpc.PUT_COMPRESSED, payload)
            return

        cur_keys: List[str] = []
        cur_blobs: List[bytes] = []
        cur_bytes = 0
        sent_hashes = False
        for k, b in zip(keys, blobs):
            entry_sz = len(k) + len(b) + 16
            if cur_keys and cur_bytes + entry_sz > max_payload:
                payload = rpc.pack_compressed(label, [] if sent_hashes else block_hashes, cur_keys, cur_blobs)
                node.rpc.call(rpc.PUT_COMPRESSED, payload)
                sent_hashes = True
                cur_keys, cur_blobs, cur_bytes = [], [], 0
            cur_keys.append(k)
            cur_blobs.append(b)
            cur_bytes += entry_sz
        if cur_keys:
            payload = rpc.pack_compressed(label, [] if sent_hashes else block_hashes, cur_keys, cur_blobs)
            node.rpc.call(rpc.PUT_COMPRESSED, payload)

    def set_bundles_priority(self, bundle_ids: List[int], priority: int) -> int:
        """Runs KVStoreLocal.set_bundles_priority on all connected daemons."""
        total = 0
        for node in self._nodes:
            total += rpc.unpack_json(node.rpc.call(rpc.SET_BUNDLES_PRIORITY, rpc.pack_json([bundle_ids, priority])))
        return total

    def sort_priority_level(self, priority: int) -> None:
        for node in self._nodes:
            node.rpc.call(rpc.SORT_PRIORITY_LEVEL, rpc.pack_json([priority]))

    def _wait(self, results: dict, layer_names, wait: bool, what: str) -> bool:
        if self.has_only_mode:
            raise RuntimeError(f"{what}() not available in has-only mode (kv_caches not provided)")
        tasks = [results[n] for n in layer_names] if layer_names else list(results.values())
        for node in self._nodes:
            node.rpc.drain()
        while not all(t.done for t in tasks):
            if not wait:
                return False
            for node in self._nodes:
                node.rpc.drain()
        return True

    def put_wait(self, put_results, layer_names=None, wait=True) -> bool:
        return self._wait(put_results, layer_names, wait, "put_wait")

    def get_wait(self, get_results, layer_names=None, wait=True) -> bool:
        return self._wait(get_results, layer_names, wait, "get_wait")

    # -- control path ------------------------------------------------------------
    def has(self, block_hashs: Optional[List[str]] = None,
            label: Optional[str] = None,
            truncate: bool = True) -> List[bool]:
        hashes = block_hashs or []
        if not hashes or len(self._nodes) == 1:
            resp = self._nodes[0].rpc.call(rpc.HAS, rpc.pack_has(hashes, label or "", truncate))
            return [bool(b) for b in resp]

        routed = self._route_hashes(hashes)
        results = [False] * len(hashes)
        for node_idx, positions in routed.items():
            sub_hashes = [hashes[p] for p in positions]
            resp = self._nodes[node_idx].rpc.call(rpc.HAS, rpc.pack_has(sub_hashes, label or "", False))
            for pos, flag in zip(positions, resp):
                results[pos] = bool(flag)

        if truncate:
            for i, ok in enumerate(results):
                if not ok:
                    for j in range(i + 1, len(results)):
                        results[j] = False
                    break
        return results

    def _json(self, method, *args):
        if len(self._nodes) == 1:
            return rpc.unpack_json(self._nodes[0].rpc.call(method, rpc.pack_json(list(args))))
        return [rpc.unpack_json(n.rpc.call(method, rpc.pack_json(list(args)))) for n in self._nodes]

    def stop(self):
        """Unregisters this client session from the daemon(s)."""
        for node in self._nodes:
            try:
                node.rpc.call(rpc.STOP)
                node.xfer.disconnect(node.peer)
            except Exception:
                logger.warning("KVStoreRemote stop failed for %s", node.peer, exc_info=True)

    def status(self) -> dict:
        res = self._json(rpc.STATUS)
        if not isinstance(res, list):
            return res
        return {
            "nodes": res,
            "current_bytes": sum(node.get("current_bytes", 0) for node in res),
            "capacity_bytes": sum(node.get("capacity_bytes", 0) for node in res),
            "group_count": sum(node.get("group_count", 0) for node in res),
        }

    def metrics(self, params: Optional[dict] = None) -> dict:
        res = self._json(rpc.METRICS, params)
        return res[0] if isinstance(res, list) else res

    def persist(self, max_count: int) -> dict:
        res = self._json(rpc.PERSIST, max_count)
        return res[0] if isinstance(res, list) else res

    def evict(self, max_count: int) -> dict:
        res = self._json(rpc.EVICT, max_count)
        return res[0] if isinstance(res, list) else res

    def get_persist_candidates(self, max_count: int) -> List[str]:
        res = self._json(rpc.PERSIST_CANDIDATES, max_count)
        return res[0] if res and isinstance(res[0], list) else res

    def get_evict_candidates(self, max_count: int) -> List[str]:
        res = self._json(rpc.EVICT_CANDIDATES, max_count)
        return res[0] if res and isinstance(res[0], list) else res
