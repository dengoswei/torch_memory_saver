import atexit
from collections import defaultdict
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
import re
import shutil
import sys
import threading
import traceback
from typing import Literal, Optional, Tuple
import weakref

import numpy as np
import torch

from .binary_wrapper import BinaryWrapper, _is_rocm_legacy_chunked
from .hooks.base import HookMode, HookUtilBase
from .utils import is_mlu, is_musa, is_xpu

logger = logging.getLogger(__name__)

_TAG_DEFAULT = "default"
BackupMode = Literal["none", "cpu", "cpu_mmap", "disk"]
CpuBackupBackend = Literal["mmap", "pinned"]
_DISK_CONFIG_LOCK = threading.Lock()
_DEVICE_POOL_META_LOCK = threading.Lock()
_DEVICE_POOL_LOCKS: dict[Tuple[int, int], threading.RLock] = {}
_ACTIVE_DEVICE_CAPTURES: dict[Tuple[int, int], int] = {}
_RETIRED_DEVICE_MEM_POOLS: dict[Tuple[int, int], list] = {}


def _get_device_pool_lock(dev_key: Tuple[int, int]) -> threading.RLock:
    with _DEVICE_POOL_META_LOCK:
        lock = _DEVICE_POOL_LOCKS.get(dev_key)
        if lock is None:
            lock = threading.RLock()
            _DEVICE_POOL_LOCKS[dev_key] = lock
        return lock


def _decrement_and_drain_device_captures(
    dev_key: Tuple[int, int], *, drain: bool = True
) -> None:
    remaining = _ACTIVE_DEVICE_CAPTURES.get(dev_key, 1) - 1
    if remaining > 0:
        _ACTIVE_DEVICE_CAPTURES[dev_key] = remaining
    else:
        _ACTIVE_DEVICE_CAPTURES.pop(dev_key, None)
        if drain:
            retired = _RETIRED_DEVICE_MEM_POOLS.pop(dev_key, None)
            if retired:
                retired.clear()
                retired = None


class _StorageMode(Enum):
    NONE = "none"
    CPU_PINNED = "pinned"
    CPU_MMAP = "mmap"
    DISK = "disk"


@dataclass(frozen=True)
class _PoolKey:
    tag: str
    device: int
    storage: _StorageMode


class _BackupTarget(Enum):
    NONE = "none"
    CPU = "cpu"
    CPU_MMAP = "cpu_mmap"
    DISK = "disk"

    @classmethod
    def from_name(cls, name: str) -> "_BackupTarget":
        if isinstance(name, str):
            for member in cls:
                if member.value == name:
                    return member
        raise ValueError(
            f"backup must be 'none', 'cpu', 'cpu_mmap', or 'disk', got {name!r}"
        )


def _validate_tag(tag: Optional[str], *, allow_none: bool) -> None:
    if tag is None:
        if allow_none:
            return
        raise ValueError("tag must be a valid UTF-8 string without NUL")
    if not isinstance(tag, str) or "\x00" in tag:
        raise ValueError("tag must be a valid UTF-8 string without NUL")
    try:
        tag.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("tag must be a valid UTF-8 string without NUL") from exc


def _validate_cpu_backup_backend_value(
    cpu_backup_backend: Optional[CpuBackupBackend],
) -> None:
    if cpu_backup_backend is not None and (
        not isinstance(cpu_backup_backend, str)
        or cpu_backup_backend not in ("pinned", "mmap")
    ):
        raise ValueError(
            f"cpu_backup_backend must be 'pinned', 'mmap', or None, got {cpu_backup_backend!r}"
        )


def _encode_utf8_path(path: str) -> bytes:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("path must be a nonempty string without NUL")
    try:
        encoded = path.encode("utf-8", errors="strict")
        os.path.realpath(encoded).decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError("path must be valid UTF-8") from exc
    return encoded


def _read_disk_backup_info(cdll) -> dict:
    return json.loads(cdll.tms_get_disk_backup_info().decode("utf-8", errors="replace"))


_PEP440_VERSION_RE = re.compile(
    r"""
    ^\s*v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?
        (?P<release>[0-9]+(?:\.[0-9]+)*)
        (?P<pre>[-_\.]?(?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)[-_\.]?(?P<pre_n>[0-9]+)?)?
        (?P<post>(?:-(?P<post_n1>[0-9]+))|(?:[-_\.]?(?P<post_l>post|rev|r)[-_\.]?(?P<post_n2>[0-9]+)?))?
        (?P<dev>[-_\.]?(?P<dev_l>dev)[-_\.]?(?P<dev_n>[0-9]+)?)?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _torch_supports_detached_mempool_free() -> bool:
    # PyTorch < 2.8.0 final selects a MemPool block's deleter via active MemPoolContext,
    # so retiring the pool before tensor free dispatches plain cudaFree (pytorch/pytorch#146431).
    match = _PEP440_VERSION_RE.fullmatch(str(getattr(torch, "__version__", "")))
    if match is None:
        return False
    epoch = int(match.group("epoch") or 0)
    release = [int(part) for part in match.group("release").split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    key = (epoch, tuple(release))
    if key > (0, (2, 8)):
        return True
    if key < (0, (2, 8)):
        return False
    if match.group("pre") is not None:
        return False
    if match.group("post") is not None:
        return True
    return match.group("dev") is None


def _resolve_region_backup(
    *,
    backup: Optional[BackupMode],
    write_once: Optional[bool],
    enable_cpu_backup: Optional[bool],
    enable_disk_backup: Optional[bool],
    readonly_disk_backup: Optional[bool],
    cpu_backup_backend: Optional[CpuBackupBackend] = None,
) -> Tuple[_BackupTarget, Optional[CpuBackupBackend], bool]:
    for flag_name, flag_val in (
        ("write_once", write_once),
        ("enable_cpu_backup", enable_cpu_backup),
        ("enable_disk_backup", enable_disk_backup),
        ("readonly_disk_backup", readonly_disk_backup),
    ):
        if flag_val is not None and not isinstance(flag_val, bool):
            raise ValueError(f"{flag_name} must be a bool, got {flag_val!r}")
    _validate_cpu_backup_backend_value(cpu_backup_backend)

    resolved_cpu_backend = cpu_backup_backend
    if backup is not None:
        raw_target = _BackupTarget.from_name(backup)
        if enable_cpu_backup is not None or enable_disk_backup is not None:
            raise ValueError(
                "backup cannot be combined with enable_cpu_backup or enable_disk_backup"
            )
        if raw_target is _BackupTarget.CPU_MMAP:
            if cpu_backup_backend is not None:
                raise ValueError("backup='cpu_mmap' cannot be combined with cpu_backup_backend")
            resolved_backup = _BackupTarget.CPU
            resolved_cpu_backend = "mmap"
        else:
            resolved_backup = raw_target
    else:
        cpu = bool(enable_cpu_backup)
        disk = bool(enable_disk_backup)
        if cpu and disk:
            raise ValueError("enable_cpu_backup and enable_disk_backup are mutually exclusive")
        resolved_backup = (
            _BackupTarget.CPU if cpu else (_BackupTarget.DISK if disk else _BackupTarget.NONE)
        )

    if write_once is not None and readonly_disk_backup is not None:
        raise ValueError("write_once cannot be combined with readonly_disk_backup")
    if readonly_disk_backup and resolved_backup is not _BackupTarget.DISK:
        raise ValueError("readonly_disk_backup requires disk backup")

    resolved_write_once = (
        bool(write_once) if write_once is not None else bool(readonly_disk_backup)
    )
    if resolved_write_once and resolved_backup is _BackupTarget.NONE:
        raise ValueError("write_once=True requires a CPU or disk backup target")
    return resolved_backup, resolved_cpu_backend, resolved_write_once


class TorchMemorySaver:
    def __init__(self):
        self._init_lock = threading.Lock()
        self._impl_ctor_kwargs: Optional[dict] = {}
        self._impl: Optional[_TorchMemorySaverImpl] = None

    @contextmanager
    def region(
        self,
        tag: str = _TAG_DEFAULT,
        enable_cpu_backup: Optional[bool] = None,
        enable_disk_backup: Optional[bool] = None,
        cpu_backup_backend: Optional[CpuBackupBackend] = None,
        readonly_disk_backup: Optional[bool] = None,
        *,
        backup: Optional[BackupMode] = None,
        write_once: Optional[bool] = None,
    ):
        """Context manager for memory saving with optional tag and backup target.

        `backup` selects where paused memory is saved: "none" (discard and
        reallocate on resume), "cpu" (pinned host shadow), "cpu_mmap"
        (reclaimable mmap host shadow), or "disk" (Linux CUDA O_DIRECT).
        Legacy aliases (`enable_cpu_backup`, `enable_disk_backup`,
        `cpu_backup_backend`, `readonly_disk_backup`) remain supported.

        Setting `write_once=True` with a CPU or disk backup target keeps the
        first pause's backup across resumes and skips D2H/disk writes on later
        pauses (intended for frozen weights; with `hook_mode="torch"` this
        requires PyTorch >= 2.8.0 final, and is unsupported on XPU, MUSA, MLU,
        and legacy chunked ROCm < 7.0).

        `backup="disk"` routes through Linux CUDA O_DIRECT NUMA-bound disk
        offload; configure a directory first via `configure_direct_disk_backup()`,
        `set_disk_backup_dir()`, or `TMS_DISK_BACKUP_DIR`.
        """
        _validate_tag(tag, allow_none=False)
        resolved_backup, resolved_cpu_backend, resolved_write_once = _resolve_region_backup(
            backup=backup,
            write_once=write_once,
            enable_cpu_backup=enable_cpu_backup,
            enable_disk_backup=enable_disk_backup,
            readonly_disk_backup=readonly_disk_backup,
            cpu_backup_backend=cpu_backup_backend,
        )
        self._validate_cpu_backup_backend(resolved_backup is _BackupTarget.CPU, resolved_cpu_backend)
        self._ensure_initialized()
        with self._impl.region(
            tag=tag,
            backup=resolved_backup,
            cpu_backup_backend=resolved_cpu_backend,
            write_once=resolved_write_once,
        ):
            yield

    @contextmanager
    def cuda_graph(
            self,
            cuda_graph, pool=None, stream=None, capture_error_mode='global',
            tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False,
            cpu_backup_backend: Optional[CpuBackupBackend] = None,
    ):
        """Similar to `torch.cuda.graph`, but ensures memory in it to be pauseable."""
        _validate_tag(tag, allow_none=False)
        if not isinstance(enable_cpu_backup, bool):
            raise ValueError(f"enable_cpu_backup must be a bool, got {enable_cpu_backup!r}")
        self._validate_cpu_backup_backend(enable_cpu_backup, cpu_backup_backend)
        self._ensure_initialized()
        with self._impl.cuda_graph(
                cuda_graph=cuda_graph,
                pool=pool, stream=stream, capture_error_mode=capture_error_mode,
                tag=tag, enable_cpu_backup=enable_cpu_backup,
                cpu_backup_backend=cpu_backup_backend,
        ):
            yield

    @contextmanager
    def disable(self):
        self._ensure_initialized()
        with self._impl.disable():
            yield

    def pause(self, tag: Optional[str] = None):
        """Pause memory for specific tag or all memory if tag is None"""
        _validate_tag(tag, allow_none=True)
        self._ensure_initialized()
        self._impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None"""
        _validate_tag(tag, allow_none=True)
        self._ensure_initialized()
        self._impl.resume(tag=tag)

    # for compatibility
    @property
    def enabled(self):
        return True

    @property
    def hook_mode(self):
        raise AttributeError

    @hook_mode.setter
    def hook_mode(self, hook_mode: HookMode):
        with self._init_lock:
            assert self._impl is None and self._impl_ctor_kwargs is not None, (
                "Cannot configure after initialization"
            )
            self._impl_ctor_kwargs["hook_mode"] = hook_mode

    @property
    def memory_margin_bytes(self):
        raise NotImplementedError("Only setter is supported")

    @memory_margin_bytes.setter
    def memory_margin_bytes(self, value: int):
        self._ensure_initialized()
        if self._impl._is_xpu:
            raise NotImplementedError(
                "TorchMemorySaver.memory_margin_bytes is not supported on Intel "
                "XPU: the OOM-margin guard needs a device-wide free-bytes "
                "reading, which the Intel GPU driver does not provide reliably "
                "(free-bytes telemetry is frozen). Manage device memory headroom "
                "outside torch_memory_saver instead."
            )
        self._impl._binary_wrapper.cdll.set_memory_margin_bytes(value)

    @property
    def retain_cpu_backup(self) -> bool:
        self._ensure_initialized()
        return self._impl._binary_wrapper.cdll.tms_get_retain_cpu_backup()

    @retain_cpu_backup.setter
    def retain_cpu_backup(self, value: bool):
        self._ensure_initialized()
        self._impl._binary_wrapper.cdll.tms_set_retain_cpu_backup(value)

    def get_cpu_backup(self, x: torch.Tensor, zero_copy: bool = False):
        self._ensure_initialized()
        return self._impl.get_cpu_backup(x, zero_copy=zero_copy)

    def configure_direct_disk_backup(
        self, path: str, *, numa_node: Optional[int] = None
    ) -> None:
        """Validate O_DIRECT reads/writes and pin one NUMA-bound staging buffer."""
        encoded_path = _encode_utf8_path(path)
        if numa_node is None:
            c_numa_node = -1
        elif isinstance(numa_node, bool) or not isinstance(numa_node, int) or not 0 <= numa_node < 1048576:
            raise ValueError("numa_node must be a nonnegative Linux NUMA node id or None")
        else:
            c_numa_node = numa_node
        self._ensure_initialized()
        with _DISK_CONFIG_LOCK:
            error = self._impl._binary_wrapper.cdll.tms_configure_direct_disk_backup(
                encoded_path, c_numa_node
            )
            if error is not None:
                raise RuntimeError(error.decode("utf-8", errors="replace"))

    def get_disk_backup_info(self) -> dict:
        """Return native capability, placement, fd flags, and cumulative I/O bytes."""
        self._ensure_initialized()
        with _DISK_CONFIG_LOCK:
            return _read_disk_backup_info(self._impl._binary_wrapper.cdll)

    def set_disk_backup_dir(self, path: str) -> None:
        """Set the directory placement for direct disk backup."""
        encoded_path = _encode_utf8_path(path)
        self._ensure_initialized()
        with _DISK_CONFIG_LOCK:
            error = self._impl._binary_wrapper.cdll.tms_set_disk_backup_dir_checked(encoded_path)
            if error is not None:
                raise RuntimeError(error.decode("utf-8", errors="replace"))

    def clear_disk_backup_dir(self, path: str) -> None:
        """Remove stale spill subdirectories under `path` before startup or offline cleanup."""
        _encode_utf8_path(path)
        if os.path.islink(path):
            raise ValueError("disk backup path must be a directory, not a symlink")
        if os.path.lexists(path) and not os.path.isdir(path):
            raise ValueError("disk backup path must be a directory")
        with _DISK_CONFIG_LOCK:
            if self._impl is not None:
                info = _read_disk_backup_info(self._impl._binary_wrapper.cdll)
                if (
                    info.get("direct")
                    and os.path.exists(path)
                    and os.path.realpath(path) == info.get("directory")
                ):
                    raise RuntimeError(
                        "cannot clear active configured disk backup directory in-process"
                    )
            if not os.path.isdir(path):
                return
            for entry in os.listdir(path):
                entry_path = os.path.join(path, entry)
                if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.unlink(entry_path)

    def _validate_cpu_backup_backend(
        self,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
    ) -> None:
        _validate_cpu_backup_backend_value(cpu_backup_backend)
        if cpu_backup_backend is None:
            return
        if not enable_cpu_backup:
            raise ValueError("cpu_backup_backend requires enable_cpu_backup=True or backup='cpu'")
        if is_xpu():
            raise ValueError(
                "cpu_backup_backend is not supported on XPU; host shadows use malloc"
            )
        if cpu_backup_backend == "mmap" and torch.version.hip:
            raise ValueError("cpu_backup_backend='mmap' is not supported on ROCm")

    def _ensure_initialized(self):
        if self._impl is not None:
            return
        with self._init_lock:
            if self._impl is not None:
                return
            assert self._impl_ctor_kwargs is not None, "Cannot configure after initialization"
            impl = _TorchMemorySaverImpl(**self._impl_ctor_kwargs)
            self._impl_ctor_kwargs = None
            self._impl = impl


_THREAD_STATE = threading.local()


class _TorchMemorySaverImpl:
    _instances: weakref.WeakSet["_TorchMemorySaverImpl"] = weakref.WeakSet()
    _region_pause_lock = threading.Lock()
    _thread_state = _THREAD_STATE
    _is_xpu: bool = False
    _is_musa: bool = False

    def _native_singleton_key(self) -> tuple:
        wrapper = getattr(self, "_binary_wrapper", None)
        resolved = getattr(wrapper, "_resolved_binary_path", None)
        if isinstance(resolved, str) and resolved:
            return ("path", resolved)
        cdll = getattr(wrapper, "cdll", None)
        if cdll is not None:
            handle = getattr(cdll, "_handle", None)
            return ("handle", handle) if handle is not None else ("cdll", id(cdll))
        return ("hook_mode", getattr(self, "_hook_mode", None))

    def __init__(self, hook_mode: HookMode = "preload"):
        self._is_xpu = is_xpu()
        self._is_musa = is_musa()
        if self._is_xpu:
            assert hook_mode == "torch", (
                "XPU only supports hook_mode='torch' (in-process pluggable "
                "allocator); preload/LD_PRELOAD is CUDA-only."
            )
        self._hook_mode = hook_mode
        self._hook_util = HookUtilBase.create(hook_mode=hook_mode)
        self._binary_wrapper = BinaryWrapper(path_binary=self._hook_util.get_path_binary())

        self._device_module = torch.get_device_module()

        if self._is_xpu:
            # Prewarm per-device SYCL contexts before registering the allocator.
            if hasattr(self._binary_wrapper.cdll, "tms_xpu_prewarm_devices"):
                self._binary_wrapper.cdll.tms_xpu_prewarm_devices(
                    self._device_module.device_count()
                )

        pool_factory = lambda: self._device_module.MemPool(allocator=self._hook_util.get_allocator())
        self._mem_pools: defaultdict[_PoolKey, object] = defaultdict(pool_factory)
        self._write_once_pools: defaultdict[_PoolKey, object] = defaultdict(pool_factory)
        if self._is_musa:
            # MusaMemPool.use() temporarily replaces the process-global
            # empty_cache, so overlapping disable() scopes are unsupported.
            self._musa_disable_lock = threading.Lock()
        self._open_write_once_counts: dict[str, int] = {}
        self._active_mem_pools: set[object] = set()
        with self._region_pause_lock:
            self._instances.add(self)

        _sanity_checks()
        if torch.version.hip or self._is_xpu:
            # HIP/SYCL runtime static dtors may run before MemPool's at exit
            # (destruction-order fiasco); free pools via atexit while runtime alive.
            atexit.register(self._clear_all_pools)

    def _clear_all_pools(self) -> None:
        self._mem_pools.clear()
        self._write_once_pools.clear()

    @contextmanager
    def _locked_region_state(self):
        class_lock = _TorchMemorySaverImpl._region_pause_lock
        inst_lock = self._region_pause_lock
        if inst_lock is not class_lock:
            with class_lock:
                with inst_lock:
                    yield
        else:
            with inst_lock:
                yield

    def _pool_map(self, write_once: bool) -> defaultdict[_PoolKey, object]:
        return self._write_once_pools if write_once else self._mem_pools

    def _reserve_active_region_pool(self, key: _PoolKey, write_once: bool):
        with self._locked_region_state():
            pools = self._pool_map(write_once)
            mem_pool = pools[key]
            if mem_pool in self._active_mem_pools:
                raise RuntimeError(
                    f"Cannot enter overlapping region() scopes sharing the same MemPool {key!r}; "
                    "use distinct tags or TorchMemorySaver instances across concurrent threads."
                )
            self._active_mem_pools.add(mem_pool)
            if write_once:
                self._open_write_once_counts[key.tag] = (
                    self._open_write_once_counts.get(key.tag, 0) + 1
                )
            return mem_pool

    def _release_active_region_pool(self, mem_pool, tag: str, write_once: bool) -> None:
        with self._locked_region_state():
            self._active_mem_pools.discard(mem_pool)
            if write_once:
                remaining = self._open_write_once_counts.get(tag, 0) - 1
                if remaining > 0:
                    self._open_write_once_counts[tag] = remaining
                else:
                    self._open_write_once_counts.pop(tag, None)

    @contextmanager
    def region(
        self,
        tag: str,
        backup: _BackupTarget,
        cpu_backup_backend: Optional[CpuBackupBackend],
        write_once: bool = False,
    ):
        _validate_tag(tag, allow_none=False)
        if not isinstance(backup, _BackupTarget) or backup is _BackupTarget.CPU_MMAP:
            raise TypeError(f"backup must be _BackupTarget (NONE, CPU, or DISK), got {backup!r}")
        if not isinstance(write_once, bool):
            raise TypeError(f"write_once must be a bool, got {write_once!r}")
        _validate_cpu_backup_backend_value(cpu_backup_backend)
        if is_mlu():
            _sanity_checks()
        if (
            sys.platform != "linux"
            or self._is_xpu
            or self._is_musa
            or is_mlu()
            or torch.version.hip
        ) and backup is _BackupTarget.DISK:
            raise NotImplementedError(
                "TorchMemorySaver direct disk backup is only supported on Linux CUDA."
            )
        if getattr(_THREAD_STATE, "active_write_once_depth", 0) > 0:
            raise RuntimeError(
                "Cannot nest a region() inside an active write_once=True region; "
                "exit the write_once region first."
            )
        cdll = self._binary_wrapper.cdll
        if write_once:
            if (
                self._is_xpu
                or self._is_musa
                or is_mlu()
                or _is_rocm_legacy_chunked(torch.version.hip)
                or not cdll.tms_supports_write_once()
            ):
                raise NotImplementedError(
                    "TorchMemorySaver write_once is not supported on this backend."
                )
            if self._hook_mode == "torch" and not _torch_supports_detached_mempool_free():
                raise NotImplementedError(
                    "TorchMemorySaver write_once with hook_mode='torch' requires "
                    "PyTorch >= 2.8.0 final (excluding 2.8.0 prereleases)."
                )
            if (
                getattr(_THREAD_STATE, "active_region_depth", 0) > 0
                or getattr(_THREAD_STATE, "active_disable_depth", 0) > 0
            ):
                raise RuntimeError(
                    "Cannot nest a write_once=True region inside an active region() or disable(); "
                    "exit the outer scope first."
                )
        # See https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
        # Key by device too: a MemPool is bound to its creation device, so a
        # multi-device process must not reuse one device's pool on another.
        device = self._current_device()
        enable_cpu_backup = backup is _BackupTarget.CPU
        enable_disk_backup = backup is _BackupTarget.DISK
        resolved_cpu_backup_backend = ""
        if enable_cpu_backup:
            resolved_cpu_backup_backend = (
                cpu_backup_backend
                or cdll.tms_get_cpu_backup_backend().decode("utf-8")
            )
        if backup is _BackupTarget.NONE:
            storage = _StorageMode.NONE
        elif backup is _BackupTarget.DISK:
            storage = _StorageMode.DISK
        elif resolved_cpu_backup_backend == "mmap":
            storage = _StorageMode.CPU_MMAP
        else:
            storage = _StorageMode.CPU_PINNED
        key = _PoolKey(tag=tag, device=device, storage=storage)
        with self._locked_region_state():
            existing_pool = self._pool_map(write_once).get(key)
            if existing_pool is not None and existing_pool in self._active_mem_pools:
                raise RuntimeError(
                    f"Cannot enter overlapping region() scopes sharing the same MemPool {key!r}; "
                    "use distinct tags or TorchMemorySaver instances across concurrent threads."
                )
        mem_pool = None
        entered_region = False
        try:
            if backup is _BackupTarget.DISK:
                with _DISK_CONFIG_LOCK:
                    disk_info = _read_disk_backup_info(cdll)
                    if not disk_info.get("direct") and not (disk_info.get("directory") or ""):
                        raise RuntimeError(
                            "configure disk backup first: call configure_direct_disk_backup(path), "
                            "set_disk_backup_dir(path), or set TMS_DISK_BACKUP_DIR"
                        )
                    mem_pool = self._reserve_active_region_pool(key, write_once)
                    entered_region = True
            else:
                mem_pool = self._reserve_active_region_pool(key, write_once)
                entered_region = True
            with self._use_thread_mem_pool(device, mem_pool):
                with self._with_region_config(
                    tag=tag,
                    enable_cpu_backup=enable_cpu_backup,
                    enable_disk_backup=enable_disk_backup,
                    cpu_backup_backend=cpu_backup_backend,
                    write_once=write_once,
                ):
                    yield
        finally:
            if entered_region:
                self._release_active_region_pool(mem_pool, tag, write_once)

    def _enter_raw_mem_pool_ctx(self, device: int, pool):
        dev_key = (id(self._device_module), int(device))
        dev_ctx_fn = getattr(self._device_module, "device", None)
        with _get_device_pool_lock(dev_key):
            ctx = None
            try:
                if callable(dev_ctx_fn) and self._current_device() != device:
                    with dev_ctx_fn(device):
                        ctx = self._device_module.use_mem_pool(pool)
                        pool = None
                        ctx.__enter__()
                else:
                    ctx = self._device_module.use_mem_pool(pool)
                    pool = None
                    ctx.__enter__()
                _ACTIVE_DEVICE_CAPTURES[dev_key] = (
                    _ACTIVE_DEVICE_CAPTURES.get(dev_key, 0) + 1
                )
                return ctx
            except BaseException as exc:
                ctx = None
                pool = None
                if exc.__traceback__ is not None:
                    traceback.clear_frames(exc.__traceback__)
                raise

    def _exit_raw_mem_pool_ctx(
        self,
        device: int,
        entry_box: list,
        exc_info,
        *,
        retire_pool: bool = False,
        drain: bool = True,
    ) -> None:
        dev_key = (id(self._device_module), int(device))
        with _get_device_pool_lock(dev_key):
            ctx = entry_box[2]
            entry_box[2] = None
            if retire_pool and entry_box[1] is not None:
                _RETIRED_DEVICE_MEM_POOLS.setdefault(dev_key, []).append(
                    entry_box[1]
                )
                entry_box[1] = None
            exited_cleanly = False
            try:
                if ctx is not None:
                    ctx.__exit__(*exc_info)
                exited_cleanly = True
            finally:
                ctx = None
                exc_info = None
                if exited_cleanly:
                    _decrement_and_drain_device_captures(dev_key, drain=drain)

    def _retire_device_mem_pool_box(self, device: int, pool_box: list) -> None:
        dev_key = (id(self._device_module), int(device))
        with _get_device_pool_lock(dev_key):
            if _ACTIVE_DEVICE_CAPTURES.get(dev_key, 0) > 0:
                if pool_box[0] is not None:
                    _RETIRED_DEVICE_MEM_POOLS.setdefault(dev_key, []).append(
                        pool_box[0]
                    )
                    pool_box[0] = None
            else:
                pool_box[0] = None

    @contextmanager
    def _use_thread_mem_pool(
        self, device: int, pool, *, retire_on_exit: bool = False
    ):
        # PyTorch routes allocations to the first matching captures_underway_ entry on the thread,
        # so suspend any outer TMS scope on this (thread, device) while keeping its MemPool alive.
        stacks = getattr(_THREAD_STATE, "active_pool_stacks", None)
        if stacks is None:
            stacks = {}
            _THREAD_STATE.active_pool_stacks = stacks
        stack = stacks.setdefault(device, [])
        outer_entry = stack[-1] if stack else None
        did_suspend_outer = outer_entry is not None and outer_entry[2] is not None
        if did_suspend_outer:
            outer_entry[0]._exit_raw_mem_pool_ctx(
                device, outer_entry, (None, None, None), retire_pool=False, drain=False
            )
        entry = [self, pool, None]
        pool = None
        try:
            entry[2] = self._enter_raw_mem_pool_ctx(device, entry[1])
        except BaseException:
            if retire_on_exit:
                fail_box = [entry[1]]
                entry[1] = None
                self._retire_device_mem_pool_box(device, fail_box)
                fail_box = None
            else:
                entry[1] = None
            if did_suspend_outer and outer_entry[2] is None:
                outer_entry[2] = outer_entry[0]._enter_raw_mem_pool_ctx(
                    device, outer_entry[1]
                )
            if not stack:
                stacks.pop(device, None)
            raise
        stack.append(entry)
        try:
            yield
        finally:
            popped = stack.pop()
            if not stack:
                stacks.pop(device, None)
            exc_info = sys.exc_info()
            try:
                self._exit_raw_mem_pool_ctx(
                    device,
                    popped,
                    exc_info,
                    retire_pool=retire_on_exit,
                )
            finally:
                exc_info = None
                if did_suspend_outer and outer_entry[2] is None:
                    outer_entry[2] = outer_entry[0]._enter_raw_mem_pool_ctx(
                        device, outer_entry[1]
                    )

    def _current_device(self) -> int:
        return self._device_module.current_device()

    @contextmanager
    def cuda_graph(
        self,
        cuda_graph,
        pool,
        stream,
        capture_error_mode,
        tag: str,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
    ):
        _validate_tag(tag, allow_none=False)
        _validate_cpu_backup_backend_value(cpu_backup_backend)
        if getattr(_THREAD_STATE, "active_write_once_depth", 0) > 0:
            raise RuntimeError(
                "Cannot capture a cuda_graph() inside an active write_once=True region; "
                "exit the write_once region first."
            )
        assert self._hook_mode == "preload", "Only hook_mode=preload supports pauseable CUDA Graph currently"
        if is_mlu():
            _sanity_checks()
        graph_ctx = self._device_module.graph(
            cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode
        )
        capture_stream = getattr(graph_ctx, "capture_stream", stream)
        graph_dev = getattr(capture_stream, "device_index", None)
        if graph_dev is None:
            graph_dev = getattr(getattr(capture_stream, "device", None), "index", None)
        if graph_dev is None:
            graph_dev = self._current_device()
        dev_key = (id(self._device_module), int(graph_dev))
        with _get_device_pool_lock(dev_key):
            _ACTIVE_DEVICE_CAPTURES[dev_key] = _ACTIVE_DEVICE_CAPTURES.get(dev_key, 0) + 1
            try:
                graph_ctx.__enter__()
            except BaseException:
                _decrement_and_drain_device_captures(dev_key)
                raise
        suppress = False
        exc_info = (None, None, None)
        try:
            with self._with_region_config(
                tag=tag,
                enable_cpu_backup=enable_cpu_backup,
                cpu_backup_backend=cpu_backup_backend,
            ):
                yield
        except BaseException:
            exc_info = sys.exc_info()
            raise
        finally:
            with _get_device_pool_lock(dev_key):
                exited_cleanly = False
                try:
                    suppress = bool(graph_ctx.__exit__(*exc_info))
                    exited_cleanly = True
                finally:
                    exc_info = (None, None, None)
                    if exited_cleanly:
                        _decrement_and_drain_device_captures(dev_key)
            if suppress:
                return

    @contextmanager
    def _with_region_config(
        self,
        tag: str,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
        enable_disk_backup: bool = False,
        write_once: bool = False,
    ):
        _validate_tag(tag, allow_none=False)
        _validate_cpu_backup_backend_value(cpu_backup_backend)
        cdll = self._binary_wrapper.cdll
        orig_tag_bytes = cdll.tms_get_current_tag()
        orig_interesting_region = cdll.tms_get_interesting_region()
        orig_enable_cpu_backup = cdll.tms_get_enable_cpu_backup()
        orig_cpu_backup_backend_bytes = cdll.tms_get_cpu_backup_backend()
        orig_enable_disk_backup = cdll.tms_get_enable_disk_backup()
        orig_write_once = cdll.tms_get_write_once()
        expected_cpu_backup_backend = (
            cpu_backup_backend or orig_cpu_backup_backend_bytes.decode("utf-8")
        )

        self._binary_wrapper.set_config(
            tag=tag,
            interesting_region=True,
            enable_cpu_backup=enable_cpu_backup,
            cpu_backup_backend=cpu_backup_backend,
            enable_disk_backup=enable_disk_backup,
            write_once=write_once,
        )

        prev_depth = getattr(_THREAD_STATE, "active_region_depth", 0)
        prev_wo_depth = getattr(_THREAD_STATE, "active_write_once_depth", 0)
        _THREAD_STATE.active_region_depth = prev_depth + 1
        if write_once:
            _THREAD_STATE.active_write_once_depth = prev_wo_depth + 1
        try:
            yield
        finally:
            _THREAD_STATE.active_region_depth = prev_depth
            _THREAD_STATE.active_write_once_depth = prev_wo_depth
            try:
                actual_interesting = cdll.tms_get_interesting_region()
                actual_cpu = cdll.tms_get_enable_cpu_backup()
                actual_cpu_backend = cdll.tms_get_cpu_backup_backend().decode("utf-8")
                actual_disk = cdll.tms_get_enable_disk_backup()
                actual_write_once = cdll.tms_get_write_once()
                actual_tag = cdll.tms_get_current_tag().decode("utf-8")
            finally:
                # Restore native TLS unconditionally before exit checks can raise AssertionError.
                cdll.tms_set_current_tag(orig_tag_bytes)
                cdll.tms_set_interesting_region(orig_interesting_region)
                cdll.tms_set_enable_cpu_backup(orig_enable_cpu_backup)
                cdll.tms_set_cpu_backup_backend(orig_cpu_backup_backend_bytes)
                cdll.tms_set_enable_disk_backup(orig_enable_disk_backup)
                cdll.tms_set_write_once(orig_write_once)
            assert actual_interesting
            assert actual_cpu == enable_cpu_backup
            assert actual_cpu_backend == expected_cpu_backup_backend
            assert actual_disk == enable_disk_backup
            assert actual_write_once == write_once
            assert actual_tag == tag

    @contextmanager
    def disable(self, dispose_mem_pool_after_use: bool = True):
        if getattr(_THREAD_STATE, "active_write_once_depth", 0) > 0:
            raise RuntimeError(
                "Cannot use TorchMemorySaver.disable() inside an active write_once=True region; "
                "exit the write_once region first."
            )
        if self._is_xpu:
            raise NotImplementedError(
                "TorchMemorySaver.disable() is not supported on Intel XPU. "
                "Allocate outside a region() instead of using disable()."
            )
        prev_disable = getattr(_THREAD_STATE, "active_disable_depth", 0)
        if self._is_musa:
            with self._musa_disable(dispose_mem_pool_after_use):
                _THREAD_STATE.active_disable_depth = prev_disable + 1
                try:
                    yield
                finally:
                    _THREAD_STATE.active_disable_depth = prev_disable
            return

        assert dispose_mem_pool_after_use, "Only dispose_mem_pool_after_use=true is supported now"
        assert self._binary_wrapper.cdll.tms_get_interesting_region(), "disable() should be called only when tms is active"

        self._binary_wrapper.cdll.tms_set_interesting_region(False)
        _THREAD_STATE.active_disable_depth = prev_disable + 1
        try:
            # We can either reuse the pool or delete it immediately, and we implement the latter currently since Slime uses it.
            # About why we need a pool: https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
            with self._use_thread_mem_pool(
                self._current_device(),
                self._device_module.MemPool(),
                retire_on_exit=True,
            ):
                yield
        finally:
            _THREAD_STATE.active_disable_depth = prev_disable
            self._binary_wrapper.cdll.tms_set_interesting_region(True)

    @contextmanager
    def _musa_disable(self, dispose_mem_pool_after_use: bool = True):
        """disable() on MUSA, where the pool needs an explicit release and
        MusaMemPool.use() temporarily replaces the process-global empty_cache."""
        from .musa_pool import MusaMemPool

        if getattr(self._thread_state, "active_region_depth", 0) > 0:
            raise RuntimeError(
                "TorchMemorySaver.disable() cannot be used inside a region() on MUSA: "
                "the two MemPool scopes cannot be nested"
            )
        # The empty_cache shim is process-global, so overlapping scopes would
        # restore it in the wrong order.
        if not self._musa_disable_lock.acquire(blocking=False):
            raise RuntimeError(
                "TorchMemorySaver.disable() cannot be nested or called concurrently on MUSA"
            )
        try:
            assert dispose_mem_pool_after_use, "Only dispose_mem_pool_after_use=true is supported now"
            assert self._binary_wrapper.cdll.tms_get_interesting_region(), "disable() should be called only when tms is active"

            self._binary_wrapper.cdll.tms_set_interesting_region(False)
            try:
                musa_pool = MusaMemPool()
                try:
                    with musa_pool.use():
                        yield
                finally:
                    musa_pool.release()
            finally:
                self._binary_wrapper.cdll.tms_set_interesting_region(True)
        finally:
            self._musa_disable_lock.release()

    def pause(self, tag: Optional[str]):
        _validate_tag(tag, allow_none=True)
        my_key = self._native_singleton_key()
        with self._region_pause_lock:
            for inst in list(self._instances):
                if inst._native_singleton_key() != my_key:
                    continue
                if any(not tag or open_tag == tag for open_tag in inst._open_write_once_counts):
                    raise RuntimeError(
                        "Cannot pause() while inside an open write_once region"
                        + (f" (tag={tag!r})" if tag else "")
                        + "; exit the region before calling pause()."
                    )
            self._sync_affected_devices(tag)
            tag_bytes = tag.encode("utf-8") if tag else None
            ret = self._binary_wrapper.cdll.tms_pause(tag_bytes)
            if self._is_xpu and ret != 0:
                raise RuntimeError(
                    f"torch_memory_saver.pause(tag={tag!r}) partially failed "
                    f"(code {ret}); some allocations could not be released and are "
                    "retained. Retry after resolving the failure."
                )
            retired_boxes = []
            for inst in list(self._instances):
                if inst._native_singleton_key() != my_key:
                    continue
                for key in list(inst._write_once_pools.keys()):
                    if not tag or key.tag == tag:
                        # Retire the write_once MemPool; destroy immediately only when no capture/pool
                        # is active on key.device, otherwise defer until active captures hit 0.
                        retired_boxes.append((inst, key.device, [inst._write_once_pools.pop(key)]))
        while retired_boxes:
            ret_inst, ret_dev, pool_box = retired_boxes.pop()
            ret_inst._retire_device_mem_pool_box(ret_dev, pool_box)
            pool_box = None

    def resume(self, tag: Optional[str]):
        _validate_tag(tag, allow_none=True)
        tag_bytes = tag.encode("utf-8") if tag else None
        ret = self._binary_wrapper.cdll.tms_resume(tag_bytes)
        if self._is_xpu:
            if ret != 0:
                raise RuntimeError(
                    f"torch_memory_saver.resume(tag={tag!r}) partially failed "
                    f"(code {ret}); some allocations remain paused. Retry after "
                    "resolving the failure (e.g. free device memory)."
                )
            self._sync_affected_devices(tag)

    def _sync_affected_devices(self, tag: Optional[str]):
        for device in self._affected_devices(tag):
            self._device_module.synchronize(device)

    def _affected_devices(self, tag: Optional[str]) -> list[int]:
        """Read tag-filtered device ids from the backend's locked allocation map."""
        device_ids = (ctypes.c_int * 64)()
        count = self._binary_wrapper.cdll.tms_affected_devices(
            tag.encode("utf-8") if tag else None,
            device_ids,
            len(device_ids),
        )
        assert count <= len(device_ids), "Too many affected devices"
        return device_ids[:count]

    def get_cpu_backup(self, x: torch.Tensor, zero_copy: bool = False):
        assert x.is_cuda or x.is_xpu or (self._is_musa and x.device.type == "musa") or (is_mlu() and x.device.type == "mlu"), f"{x.device=}"
        assert x.is_contiguous(), f"{x.shape=} {x.stride()=} {x.dtype=}"

        nbytes = x.nbytes
        gpu_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_uint8))
        cpu_ptr = self._binary_wrapper.cdll.tms_get_cpu_backup_pointer(gpu_ptr, nbytes)
        if not cpu_ptr:
            return None

        np_untyped = np.ctypeslib.as_array(cpu_ptr, shape=(nbytes,))
        assert np_untyped.dtype == np.uint8, f"{np_untyped.dtype=} {np_untyped.shape=}"

        ans_untyped = torch.from_numpy(np_untyped)
        ans = ans_untyped.view(x.dtype).view(x.shape)

        # For simplicity and safety
        if not zero_copy:
            ans = ans.clone()

        assert ans.device == torch.device("cpu"), f"{ans.device=}"
        assert ans.dtype == x.dtype, f"{ans.dtype=} {x.dtype=}"
        assert ans.shape == x.shape, f"{ans.shape=} {x.shape=}"
        assert ans.stride() == x.stride(), f"{ans.stride()=} {x.stride()=}"
        return ans

def _sanity_checks():
    if is_mlu():
        expandable_segments = torch.mlu.memory._snapshot()["allocator_settings"]["expandable_segments"]
    else:
        alloc_confs = ["PYTORCH_CUDA_ALLOC_CONF"]
        if is_musa():
            alloc_confs.append("PYTORCH_MUSA_ALLOC_CONF")
        expandable_segments = any("expandable_segments:True" in os.environ.get(name, "") for name in alloc_confs)
    if expandable_segments:
        raise RuntimeError(
            "TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet."
        )
