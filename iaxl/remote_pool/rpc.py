"""RPC between KVStoreRemote (client) and the daemon's KVStoreLocal, over one NIXL agent.

Transport (§3.8 of DESIGN.md): every side owns a registered control buffer.
Client requests RDMA-WRITE their payload into the daemon's buffer and carry the
header in the notification attached to that write, so the header arrives after
the data has landed. Responses and `done` pushes are plain notifications.

  request  header  "<cBIQ"  kind (b"Q": payload in daemon buf | b"I": inline), method, seq, len
  response header  "<cIB"   b"R", seq, ok            + payload
  push     header  "<cIH"   b"D", job_id, layer_idx
"""

import json
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import torch

from ..utils import cuda_available

logger = logging.getLogger(__name__)

CTRL_BYTES = 4 << 20

HELLO, REGISTER_KV_CACHES, REGISTER_LAYERS, PUT, GET, HAS, STATUS, METRICS, PERSIST, EVICT, \
    PERSIST_CANDIDATES, EVICT_CANDIDATES, STOP, UNREGISTER, SET_BUNDLES_PRIORITY, \
    SORT_PRIORITY_LEVEL, PUT_COMPRESSED = range(17)

REQ_FMT, RESP_FMT, PUSH_FMT = "<cBIQ", "<cIB", "<cIH"
REQ_SIZE, RESP_SIZE, PUSH_SIZE = (struct.calcsize(f) for f in (REQ_FMT, RESP_FMT, PUSH_FMT))


def rank_port(port: int, rank: int) -> int:
    return port + 1 + rank


# -- codec ---------------------------------------------------------------------

def pack_hashes(hashes: List[str]) -> bytes:
    """Fixed-width ASCII records (width = longest hash, NUL padded)."""
    if not hashes:
        return struct.pack("<IH", 0, 0)
    h = max(map(len, hashes))
    return struct.pack("<IH", len(hashes), h) + np.array(hashes, dtype=f"S{h}").tobytes()


def unpack_hashes(buf: bytes, off: int = 0):
    n, h = struct.unpack_from("<IH", buf, off)
    off += 6
    if n == 0:
        return [], off
    hashes = np.frombuffer(buf, f"S{h}", count=n, offset=off).astype(str).tolist()
    return hashes, off + n * h


def pack_blocks(block_indices, block_hashs, layer_idx, description: str, label: str = "",
                priorities: Optional[List[int]] = None, bundle_id: int = 0) -> bytes:
    """`priorities`: one OMP-Main level per entry of `layer_idx` (GET only; None: level 0)."""
    desc = description.encode()
    lab = label.encode()
    levels = priorities or []
    return b"".join((
        struct.pack("<IHHHHQ", len(block_indices), len(layer_idx), len(levels), len(desc),
                    len(lab), bundle_id),
        np.asarray(block_indices, dtype=np.int32).tobytes(),
        np.asarray(layer_idx, dtype=np.int16).tobytes(),
        np.asarray(levels, dtype=np.int8).tobytes(),
        pack_hashes(block_hashs),
        desc,
        lab,
    ))


def unpack_blocks(buf: bytes):
    n, nl, npri, nd, nlab, bundle_id = struct.unpack_from("<IHHHHQ", buf)
    off = 20
    indices = np.frombuffer(buf, np.int32, count=n, offset=off).tolist()
    off += 4 * n
    layer_idx = np.frombuffer(buf, np.int16, count=nl, offset=off).tolist()
    off += 2 * nl
    priorities = np.frombuffer(buf, np.int8, count=npri, offset=off).tolist()
    off += npri
    hashes, off = unpack_hashes(buf, off)
    desc = buf[off:off + nd].decode()
    off += nd
    return indices, hashes, layer_idx, desc, buf[off:off + nlab].decode(), priorities, bundle_id


def pack_has(block_hashs: List[str], label: str = "", truncate: bool = True) -> bytes:
    lab = label.encode()
    return b"".join((
        struct.pack("<BB", len(lab), truncate),
        lab,
        pack_hashes(block_hashs),
    ))


def unpack_has(buf: bytes):
    nlab, truncate = struct.unpack_from("<BB", buf)
    off = 2
    label = buf[off:off + nlab].decode()
    off += nlab
    hashes, off = unpack_hashes(buf, off)
    return hashes, label, bool(truncate)


def pack_json(obj) -> bytes:
    return json.dumps(obj).encode()


def unpack_json(buf: bytes):
    return json.loads(buf) if buf else None


def pack_compressed(label: str, block_hashes: List[str], keys: List[str], blobs: List[bytes]) -> bytes:
    """Pack compressed KV blobs for Host-to-Host RDMA spill (PUT_COMPRESSED)."""
    lab = label.encode()
    parts = [struct.pack("<HI", len(lab), len(keys)), lab, pack_hashes(block_hashes)]
    for k, b in zip(keys, blobs):
        kb = k.encode()
        parts.append(struct.pack("<HI", len(kb), len(b)))
        parts.append(kb)
        parts.append(b)
    return b"".join(parts)


def unpack_compressed(buf: bytes):
    nlab, nkeys = struct.unpack_from("<HI", buf, 0)
    off = 6
    label = buf[off:off + nlab].decode()
    off += nlab
    block_hashes, off = unpack_hashes(buf, off)
    keys: List[str] = []
    blobs: List[bytes] = []
    for _ in range(nkeys):
        nk, nb = struct.unpack_from("<HI", buf, off)
        off += 6
        keys.append(buf[off:off + nk].decode())
        off += nk
        blobs.append(buf[off:off + nb])
        off += nb
    return label, block_hashes, keys, blobs


def _ctrl_buffer(pin: bool) -> torch.Tensor:
    return torch.empty(CTRL_BYTES, dtype=torch.uint8, device="cpu", pin_memory=pin)


def _json_call(name: str):
    """Handler forwarding JSON-encoded positional args to `KVStore.<name>`."""
    def handler(self, peer, payload):
        args = unpack_json(payload) or []
        return pack_json(getattr(self.kvstore, name)(*args))
    return handler


# -- client --------------------------------------------------------------------

class RpcChannel:
    """Synchronous, single-outstanding client channel. `on_push(job_id, layer_idx)`
    receives `done` pushes that arrive while waiting for responses or in drain()."""

    def __init__(self, xfer, peer: str, on_push: Callable[[int, int], None]):
        self.xfer, self.peer, self.on_push = xfer, peer, on_push
        self.buf = _ctrl_buffer(pin=True)
        self.buf_np = self.buf.numpy()
        self.buf_ptr = self.buf.data_ptr()
        xfer.register_memory(self.buf, "rpc.buf")
        self.remote_base = 0
        self.remote_bytes = 0
        self._seq = 0
        self._resp = None
        self._lock = threading.Lock()

    def handshake(self):
        info = unpack_json(self.call(HELLO))
        self.remote_base, self.remote_bytes = info["base"], info["bytes"]

    def call(self, method: int, payload: bytes = b"") -> bytes:
        with self._lock:
            self._seq = seq = (self._seq + 1) & 0xFFFFFFFF
            n = len(payload)
            if 0 < n <= self.remote_bytes:
                self.buf_np[:n] = np.frombuffer(payload, np.uint8)
                self.xfer.write(self.peer, self.buf_ptr, self.remote_base, n,
                                struct.pack(REQ_FMT, b"Q", method, seq, n))
            else:
                self.xfer.send_notif(self.peer, struct.pack(REQ_FMT, b"I", method, seq, n) + payload)
            while self._resp is None:
                self.drain()
            rseq, ok, body = self._resp
            self._resp = None
            assert rseq == seq, f"rpc: response seq {rseq} != {seq}"
            if not ok:
                raise RuntimeError(f"rpc method {method} failed on {self.peer}: {body.decode(errors='replace')}")
            return body

    def drain(self):
        for _, msg in self.xfer.iter_notifs():
            kind = msg[:1]
            if kind == b"D":
                _, job_id, layer_idx = struct.unpack_from(PUSH_FMT, msg)
                self.on_push(job_id, layer_idx)
            elif kind == b"R":
                _, seq, ok = struct.unpack_from(RESP_FMT, msg)
                self._resp = (seq, ok, msg[RESP_SIZE:])
            else:
                logger.warning("rpc: unexpected notif kind %r", kind)


# -- daemon --------------------------------------------------------------------

@dataclass
class Job:
    peer: str
    tasks: dict
    is_put: bool
    not_done: Set[str] = field(default_factory=set)


@dataclass
class ClientSession:
    peer: str
    client_slot: int
    ctrl: torch.Tensor
    ctrl_np: np.ndarray
    ctrl_pool_name: str
    remote_names: List[str] = field(default_factory=list)
    pool_keys: Dict[str, str] = field(default_factory=dict)


class KVStoreService:
    """Daemon-side handlers. Owns a shared KVStoreLocal and per-client sessions
    (each client gets its own registered control buffer and RemoteTensor pools)."""

    def __init__(self, xfer, role: str, rank: int = 0, tp_size: int = 1):
        self.xfer, self.role, self.rank, self.tp_size = xfer, role, rank, tp_size
        self.kvstore = None
        self.layer_names: List[str] = []
        self.layer_idx: Dict[str, int] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self._next_slot = 0
        self.jobs: Dict[int, Job] = {}
        self._next_job = 0

        self._max_slots = 16
        self._ctrl_slots: List[tuple[torch.Tensor, np.ndarray]] = []
        self._free_slots: List[int] = []
        pin = cuda_available()
        for slot in range(self._max_slots):
            ctrl = _ctrl_buffer(pin=pin)
            self.xfer.register_memory(ctrl, f"rpc.ctrl.slot{slot}")
            self._ctrl_slots.append((ctrl, ctrl.numpy()))
            self._free_slots.append(slot)

    def ctrl_buf_for(self, peer: str) -> np.ndarray:
        sess = self.sessions.get(peer)
        if sess is None:
            raise KeyError(f"unknown peer {peer} (handshake required)")
        return sess.ctrl_np

    # -- dispatch ------------------------------------------------------------
    def dispatch(self, peer: str, method: int, payload: bytes) -> bytes:
        return self._HANDLERS[method](self, peer, payload)

    def _cleanup_session(self, peer: str):
        sess = self.sessions.pop(peer, None)
        if sess is None:
            return
        if sess.remote_names:
            from iaxl import torch_ext
            for pname in sess.remote_names:
                torch_ext.unregister_kv_pool(pname)
            sess.remote_names.clear()
        if sess.client_slot not in self._free_slots:
            self._free_slots.append(sess.client_slot)
        for job_id in [jid for jid, j in self.jobs.items() if j.peer == peer]:
            del self.jobs[job_id]

    def _hello(self, peer, payload):
        self.xfer.wait_peer(peer)  # we need the client's metadata to notify it
        if peer in self.sessions:
            self._cleanup_session(peer)
        if not self._free_slots:
            raise RuntimeError(f"max client sessions ({self._max_slots}) reached")
        slot = self._free_slots.pop(0)
        ctrl, ctrl_np = self._ctrl_slots[slot]
        sess = ClientSession(
            peer=peer,
            client_slot=slot,
            ctrl=ctrl,
            ctrl_np=ctrl_np,
            ctrl_pool_name=f"rpc.ctrl.slot{slot}",
        )
        self.sessions[peer] = sess
        return pack_json({"base": ctrl.data_ptr(), "bytes": CTRL_BYTES, "client_slot": slot})

    def _check_topology(self, req):
        if req["tp_size"] != self.tp_size:
            raise ValueError(f"client tp_size {req['tp_size']} != daemon tp_size {self.tp_size}")
        if req.get("rank", 0) != self.rank:
            raise ValueError(f"client rank {req.get('rank')} != daemon rank {self.rank}")

    def _register_kv_caches(self, peer, payload):
        from iaxl import torch_ext
        from iaxl.kvstore import KVStoreLocal

        from .remote_tensor import RemoteTensor

        req = unpack_json(payload)
        self._check_topology(req)
        sess = self.sessions[peer]
        block_dim = req["block_dim"]
        kv_caches = {}
        for name, t in req["layers"].items():
            dtype = getattr(torch, t["dtype"])
            shape = tuple(t["shape"])
            pool_name = f"{peer}:{name}"
            rt = RemoteTensor(peer, t["base"], shape, dtype, t["dev_id"])
            # Per-peer Remote pool so multiple HeadNodes never collide on KVPool names.
            torch_ext.register_kv_pool_remote(pool_name, peer, rt.base, list(shape), rt.element_size(),
                                              dtype, rt.dev_id, block_dim)
            sess.remote_names.append(pool_name)
            sess.pool_keys[name] = pool_name
            kv_caches[name] = rt
        if self.kvstore is None:
            self.kvstore = KVStoreLocal(model_name=req["model_name"], block_dim=block_dim,
                                        kv_caches=kv_caches, rank=self.rank,
                                        tp_size=self.tp_size, shared_mode=True)
            self._set_layers(self.kvstore.layer_names)
        return pack_json({"layer_names": self.layer_names})

    def _register_layers(self, peer, payload):
        from iaxl.kvstore import KVStoreLocal

        req = unpack_json(payload)
        self._check_topology(req)
        if self.kvstore is None:
            self.kvstore = KVStoreLocal(model_name=req["model_name"], layer_names=req["layer_names"],
                                        tp_size=self.tp_size, shared_mode=True)
            self._set_layers(self.kvstore.layer_names)
        return b""

    def _set_layers(self, names):
        self.layer_names = list(names)
        self.layer_idx = {n: i for i, n in enumerate(self.layer_names)}

    def _xfer(self, peer, payload, is_put):
        indices, hashes, layer_idx, desc, label, priorities, bundle_id = unpack_blocks(payload)
        names = [self.layer_names[i] for i in layer_idx] or None
        sess = self.sessions[peer]
        if is_put:
            tasks = self.kvstore.put(indices, hashes, names, desc, label=label or None,
                                     pool_keys=sess.pool_keys)
        else:
            layer_priorities = None
            if priorities:
                if len(priorities) != len(layer_idx):
                    raise ValueError(f"GET has {len(priorities)} priorities for {len(layer_idx)} layers")
                layer_priorities = {}
                for name, priority in zip(names, priorities):
                    layer_priorities[name] = priority
            tasks = self.kvstore.get(indices, hashes, names, desc, label=label or None,
                                     priorities=layer_priorities, bundle_id=bundle_id,
                                     pool_keys=sess.pool_keys)
        self._next_job = job_id = (self._next_job + 1) & 0xFFFFFFFF
        self.jobs[job_id] = Job(peer, tasks, is_put, set(tasks.keys()))
        return struct.pack("<I", job_id)

    def _put(self, peer, payload):
        return self._xfer(peer, payload, True)

    def _get(self, peer, payload):
        return self._xfer(peer, payload, False)

    def _put_compressed(self, peer, payload):
        label, block_hashes, keys, blobs = unpack_compressed(payload)
        sess = self.sessions[peer]
        self.kvstore.import_compressed(label, block_hashes, keys, blobs,
                                       client_slot=sess.client_slot)
        return b""

    def _has(self, peer, payload):
        hashes, label, truncate = unpack_has(payload)
        sess = self.sessions[peer]
        flags = self.kvstore.has(hashes, label=label or None, truncate=truncate,
                                 client_slot=sess.client_slot)
        return np.asarray(flags, dtype=np.uint8).tobytes()

    def _stop(self, peer, payload):
        self._cleanup_session(peer)
        req = unpack_json(payload) or {}
        if req.get("force") and self.kvstore is not None:
            self.kvstore.stop()
            return b"STOP"
        return b""

    def _unregister(self, peer, payload):
        self._cleanup_session(peer)
        return b""

    _HANDLERS = {
        HELLO: _hello,
        REGISTER_KV_CACHES: _register_kv_caches,
        REGISTER_LAYERS: _register_layers,
        PUT: _put,
        GET: _get,
        HAS: _has,
        STATUS: _json_call("status"),
        METRICS: _json_call("metrics"),
        PERSIST: _json_call("persist"),
        EVICT: _json_call("evict"),
        PERSIST_CANDIDATES: _json_call("get_persist_candidates"),
        EVICT_CANDIDATES: _json_call("get_evict_candidates"),
        STOP: _stop,
        UNREGISTER: _unregister,
        SET_BUNDLES_PRIORITY: _json_call("set_bundles_priority"),
        SORT_PRIORITY_LEVEL: _json_call("sort_priority_level"),
        PUT_COMPRESSED: _put_compressed,
    }

    # -- completion polling --------------------------------------------------
    def poll_done(self):
        """Yield (job_id, layer_idx, peer) for every layer that completed since the
        last call; completed layers release their pool blocks immediately."""
        kv = self.kvstore
        for job_id in list(self.jobs):
            job = self.jobs[job_id]
            wait = kv.put_wait if job.is_put else kv.get_wait
            for name in list(job.not_done):
                if wait(job.tasks, [name], wait=False):
                    job.not_done.discard(name)
                    yield job_id, self.layer_idx[name], job.peer
            if not job.not_done:
                del self.jobs[job_id]


def serve(xfer, service: KVStoreService, idle_spin: int = 2000, idle_sleep: float = 50e-6):
    """Single-threaded daemon loop: dispatch requests, poll completions, push `done`."""
    idle = 0
    while True:
        busy = False
        for peer, msg in xfer.iter_notifs():
            busy = True
            kind, method, seq, n = struct.unpack_from(REQ_FMT, msg)
            payload = (
                service.ctrl_buf_for(peer)[:n].tobytes()
                if kind == b"Q"
                else msg[REQ_SIZE:REQ_SIZE + n]
            )
            try:
                ok, body = True, service.dispatch(peer, method, payload)
            except Exception as e:  # report to the caller, keep serving
                logger.exception("rpc: method %d from %s failed", method, peer)
                ok, body = False, str(e).encode()
            xfer.send_notif(peer, struct.pack(RESP_FMT, b"R", seq, ok) + body)
            if method == STOP and ok and body == b"STOP":
                return
        for job_id, layer_idx, peer in service.poll_done():
            busy = True
            xfer.send_notif(peer, struct.pack(PUSH_FMT, b"D", job_id, layer_idx))
        if busy or service.jobs:
            idle = 0
        elif idle < idle_spin:
            idle += 1
        else:
            time.sleep(idle_sleep)
