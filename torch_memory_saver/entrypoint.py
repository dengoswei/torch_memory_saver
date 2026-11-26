import atexit
import ctypes
import json

import numpy as np
import logging
import os
import re
import shutil
import sys
import threading
import weakref
from collections import defaultdict
from contextlib import contextmanager
from typing import Literal, Optional, Tuple
import torch
from torch.torch_version import TorchVersion

from .binary_wrapper import BinaryWrapper
from .hooks.base import HookUtilBase, HookMode
from .utils import is_mlu, is_musa, is_xpu

logger = logging.getLogger(__name__)

_TAG_DEFAULT = "default"
BackupMode = Literal["none", "cpu", "cpu_mmap", "disk"]
CpuBackupBackend = Literal["mmap", "pinned"]
_DISK_CONFIG_LOCK = threading.Lock()
_BACKUP_MODES: dict[str, Tuple[bool, bool, Optional[CpuBackupBackend]]] = {
    "none": (False, False, None),
    "cpu": (True, False, None),
    "cpu_mmap": (True, False, "mmap"),
    "disk": (False, True, None),
}


def _validate_tag(tag: str) -> None:
    if not isinstance(tag, str) or "\x00" in tag:
        raise ValueError("tag must be a valid UTF-8 string without NUL")
    try:
        tag.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("tag must be a valid UTF-8 string without NUL") from exc


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


_SPILL_DIR_RE = re.compile(r"^pid_\d+(?:_[A-Za-z0-9]+)?\Z")


def _torch_supports_detached_mempool_free() -> bool:
    # PyTorch < 2.8.0 final selects a MemPool block's deleter via active MemPoolContext,
    # so retiring the pool before tensor free dispatches plain cudaFree (pytorch/pytorch#146431).
    try:
        return bool(TorchVersion(str(getattr(torch, "__version__", ""))) >= (2, 8, 0))
    except TypeError:
        return False


def _validate_region_flags(
    *,
    write_once: bool,
    enable_cpu_backup: Optional[bool],
    enable_disk_backup: Optional[bool],
) -> None:
    if not isinstance(write_once, bool):
        raise ValueError(f"write_once must be a bool, got {write_once!r}")
    for flag_name, flag_val in (
        ("enable_cpu_backup", enable_cpu_backup),
        ("enable_disk_backup", enable_disk_backup),
    ):
        if flag_val is not None and not isinstance(flag_val, bool):
            raise ValueError(f"{flag_name} must be a bool, got {flag_val!r}")


def _resolve_explicit_backup(
    backup: BackupMode,
    enable_cpu_backup: Optional[bool],
    enable_disk_backup: Optional[bool],
    cpu_backup_backend: Optional[CpuBackupBackend],
) -> Tuple[bool, bool, Optional[CpuBackupBackend]]:
    if not isinstance(backup, str) or backup not in _BACKUP_MODES:
        raise ValueError(
            f"backup must be 'none', 'cpu', 'cpu_mmap', or 'disk', got {backup!r}"
        )
    if enable_cpu_backup is not None or enable_disk_backup is not None:
        raise ValueError(
            "backup cannot be combined with enable_cpu_backup or enable_disk_backup"
        )
    if backup == "cpu_mmap" and cpu_backup_backend is not None:
        raise ValueError("backup='cpu_mmap' cannot be combined with cpu_backup_backend")
    cpu, disk, forced_backend = _BACKUP_MODES[backup]
    return cpu, disk, forced_backend or cpu_backup_backend


def _resolve_region_backup(
    *,
    backup: Optional[BackupMode],
    write_once: bool,
    enable_cpu_backup: Optional[bool],
    enable_disk_backup: Optional[bool],
    cpu_backup_backend: Optional[CpuBackupBackend] = None,
) -> Tuple[bool, bool, Optional[CpuBackupBackend]]:
    _validate_region_flags(
        write_once=write_once,
        enable_cpu_backup=enable_cpu_backup,
        enable_disk_backup=enable_disk_backup,
    )
    if backup is not None:
        cpu, disk, resolved_backend = _resolve_explicit_backup(
            backup, enable_cpu_backup, enable_disk_backup, cpu_backup_backend
        )
    else:
        cpu = bool(enable_cpu_backup)
        disk = bool(enable_disk_backup)
        if cpu and disk:
            raise ValueError("enable_cpu_backup and enable_disk_backup are mutually exclusive")
        resolved_backend = cpu_backup_backend
    if write_once and not cpu and not disk:
        raise ValueError("write_once=True requires a CPU or disk backup target")
    return cpu, disk, resolved_backend


def _require_disk_configured(cdll) -> None:
    with _DISK_CONFIG_LOCK:
        disk_info = _read_disk_backup_info(cdll)
        if not disk_info.get("direct") and not (disk_info.get("directory") or ""):
            raise RuntimeError(
                "configure disk backup first: call configure_direct_disk_backup(path), "
                "set_disk_backup_dir(path), or set TMS_DISK_BACKUP_DIR"
            )


def _remove_stale_spill_dirs(path: str) -> None:
    if not os.path.isdir(path):
        return
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        if _SPILL_DIR_RE.match(entry) and os.path.isdir(entry_path) and not os.path.islink(entry_path):
            shutil.rmtree(entry_path)


class TorchMemorySaver:
    def __init__(self):
        self._impl_ctor_kwargs = {}
        self._impl: Optional[_TorchMemorySaverImpl] = None

    @contextmanager
    def region(
        self,
        tag: str = _TAG_DEFAULT,
        enable_cpu_backup: Optional[bool] = None,
        enable_disk_backup: Optional[bool] = None,
        cpu_backup_backend: Optional[CpuBackupBackend] = None,
        *,
        backup: Optional[BackupMode] = None,
        write_once: bool = False,
    ):
        """Context manager for memory saving with optional tag and backup target.

        `backup` selects where paused memory is saved: "none" (discard and
        reallocate on resume), "cpu" (host shadow; "pinned" by default, or
        "mmap" via `cpu_backup_backend`/`TMS_INIT_CPU_BACKUP_BACKEND`),
        "cpu_mmap" (reclaimable mmap host shadow), or "disk" (Linux CUDA O_DIRECT).
        Legacy aliases `enable_cpu_backup` and `cpu_backup_backend` match master;
        `enable_disk_backup=True` aliases `backup="disk"` and is likewise Linux
        CUDA O_DIRECT-only (unsupported on ROCm, XPU, MUSA, and MLU).

        Setting `write_once=True` with a CPU or disk backup target keeps the
        first pause's backup across resumes and skips D2H/disk writes on later
        pauses (intended for frozen weights; with `hook_mode="torch"` this
        requires PyTorch >= 2.8.0 final, and is unsupported on XPU, MUSA, MLU,
        and legacy chunked ROCm < 7.0).

        `backup="disk"` (or `enable_disk_backup=True`) routes through Linux CUDA
        O_DIRECT NUMA-bound disk offload; configure a directory first via
        `configure_direct_disk_backup()`, `set_disk_backup_dir()`, or
        `TMS_DISK_BACKUP_DIR`.
        """
        _validate_tag(tag)
        enable_cpu_backup, enable_disk_backup, resolved_cpu_backend = _resolve_region_backup(
            backup=backup,
            write_once=write_once,
            enable_cpu_backup=enable_cpu_backup,
            enable_disk_backup=enable_disk_backup,
            cpu_backup_backend=cpu_backup_backend,
        )
        self._validate_cpu_backup_backend(enable_cpu_backup, resolved_cpu_backend)
        self._ensure_initialized()
        with self._impl.region(
            tag=tag,
            enable_cpu_backup=enable_cpu_backup,
            enable_disk_backup=enable_disk_backup,
            cpu_backup_backend=resolved_cpu_backend,
            write_once=write_once,
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
        self._ensure_initialized()
        self._impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None"""
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
        assert self._impl_ctor_kwargs is not None, "Cannot configure after initialization"
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

    def _has_active_disk_dir(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        real_target = os.path.realpath(path)
        for wrapper in _TorchMemorySaverImpl._loaded_wrappers.values():
            info = _read_disk_backup_info(wrapper.cdll)
            if info.get("direct") and real_target == info.get("directory"):
                return True
        return False

    def clear_disk_backup_dir(self, path: str) -> None:
        """Offline cleanup: remove stale pid_* spill subdirectories when no other process uses `path`."""
        _encode_utf8_path(path)
        if os.path.islink(path):
            raise ValueError("disk backup path must be a directory, not a symlink")
        if os.path.lexists(path) and not os.path.isdir(path):
            raise ValueError("disk backup path must be a directory")
        with _DISK_CONFIG_LOCK:
            if self._has_active_disk_dir(path):
                raise RuntimeError(
                    "cannot clear active configured disk backup directory in-process"
                )
            _remove_stale_spill_dirs(path)

    def _validate_cpu_backup_backend(
        self,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
    ) -> None:
        if cpu_backup_backend is None:
            return
        if not isinstance(cpu_backup_backend, str) or cpu_backup_backend not in ("pinned", "mmap"):
            raise ValueError(
                f"cpu_backup_backend must be 'pinned', 'mmap', or None, got {cpu_backup_backend!r}"
            )
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
        self._impl = _TorchMemorySaverImpl(**self._impl_ctor_kwargs)
        del self._impl_ctor_kwargs


_THREAD_STATE = threading.local()


class _TorchMemorySaverImpl:
    _instances: weakref.WeakSet["_TorchMemorySaverImpl"] = weakref.WeakSet()
    _loaded_wrappers: dict[str, BinaryWrapper] = {}
    _region_pause_lock = threading.Lock()
    _active_scopes: int = 0
    _thread_state = _THREAD_STATE
    _is_xpu: bool = False
    _is_musa: bool = False

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
        self._mem_pools: defaultdict = defaultdict(pool_factory)
        self._write_once_pools: defaultdict = defaultdict(pool_factory)
        if self._is_musa:
            # MusaMemPool.use() temporarily replaces the process-global
            # empty_cache, so overlapping disable() scopes are unsupported.
            self._musa_disable_lock = threading.Lock()
        self._active_mem_pools: set[object] = set()
        with _DISK_CONFIG_LOCK:
            self._loaded_wrappers[self._binary_wrapper._resolved_binary_path] = self._binary_wrapper
        with _TorchMemorySaverImpl._region_pause_lock:
            self._instances.add(self)

        _sanity_checks()
        if torch.version.hip or self._is_xpu:
            # HIP/SYCL runtime static dtors may run before MemPool's at exit
            # (destruction-order fiasco); free pools via atexit while runtime alive.
            atexit.register(self._clear_all_pools)

    def _clear_all_pools(self) -> None:
        self._mem_pools.clear()
        self._write_once_pools.clear()

    def _check_disk_capabilities(self) -> None:
        if (
            sys.platform != "linux"
            or self._is_xpu
            or self._is_musa
            or is_mlu()
            or torch.version.hip
        ):
            raise NotImplementedError(
                "TorchMemorySaver direct disk backup is only supported on Linux CUDA."
            )

    def _check_write_once_capabilities(self) -> None:
        if not self._binary_wrapper.cdll.tms_supports_write_once():
            raise NotImplementedError(
                "TorchMemorySaver write_once is not supported on this backend."
            )
        if self._hook_mode == "torch" and not _torch_supports_detached_mempool_free():
            raise NotImplementedError(
                "TorchMemorySaver write_once with hook_mode='torch' requires "
                "PyTorch >= 2.8.0 final (excluding 2.8.0 prereleases)."
            )

    def _check_region_capabilities(self, enable_disk_backup: bool, write_once: bool) -> None:
        if enable_disk_backup:
            self._check_disk_capabilities()
        if getattr(_THREAD_STATE, "active_write_once_depth", 0) > 0:
            raise RuntimeError(
                "Cannot nest a region() inside an active write_once=True region; "
                "exit the write_once region first."
            )
        if write_once:
            self._check_write_once_capabilities()
        if write_once and (
            getattr(_THREAD_STATE, "active_region_depth", 0) > 0
            or getattr(_THREAD_STATE, "active_disable_depth", 0) > 0
        ):
            raise RuntimeError(
                "Cannot nest a write_once=True region inside an active region() or disable(); "
                "exit the outer scope first."
            )

    @contextmanager
    def region(
        self,
        tag: str,
        enable_cpu_backup: bool,
        enable_disk_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
        write_once: bool = False,
    ):
        if is_mlu():
            _sanity_checks()
        self._check_region_capabilities(enable_disk_backup, write_once)
        cdll = self._binary_wrapper.cdll
        if enable_disk_backup:
            _require_disk_configured(cdll)

        # See https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
        # Key by device too: a MemPool is bound to its creation device, so a
        # multi-device process must not reuse one device's pool on another.
        device = self._current_device()
        resolved_cpu_backup_backend = ""
        if enable_cpu_backup:
            resolved_cpu_backup_backend = (
                cpu_backup_backend or cdll.tms_get_cpu_backup_backend().decode("utf-8")
            )
        key = (
            tag,
            enable_cpu_backup,
            enable_disk_backup,
            resolved_cpu_backup_backend,
            device,
        )
        with _TorchMemorySaverImpl._region_pause_lock:
            pools = self._write_once_pools if write_once else self._mem_pools
            mem_pool = pools[key]
            if mem_pool in self._active_mem_pools:
                raise RuntimeError(
                    f"Cannot enter overlapping region() scopes sharing the same MemPool {key!r}; "
                    "use distinct tags or TorchMemorySaver instances across concurrent threads."
                )
            self._active_mem_pools.add(mem_pool)
            _TorchMemorySaverImpl._active_scopes += 1
        try:
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
            with _TorchMemorySaverImpl._region_pause_lock:
                _TorchMemorySaverImpl._active_scopes -= 1
                self._active_mem_pools.discard(mem_pool)
                mem_pool = None

    def _enter_pool_ctx(self, device: int, pool):
        dev_ctx_fn = getattr(self._device_module, "device", None)
        if callable(dev_ctx_fn) and self._current_device() != device:
            with dev_ctx_fn(device):
                ctx = self._device_module.use_mem_pool(pool)
                ctx.__enter__()
                return ctx
        ctx = self._device_module.use_mem_pool(pool)
        ctx.__enter__()
        return ctx

    @contextmanager
    def _suspended_outer_pool(self, device: int):
        stacks = _THREAD_STATE.__dict__.setdefault("active_pool_stacks", {})
        stack = stacks.setdefault(device, [])
        outer = stack[-1] if stack else None
        if outer is not None:
            outer[1].__exit__(None, None, None)
            outer[1] = None
        try:
            yield stack
        finally:
            if outer is not None:
                outer[1] = outer[2]._enter_pool_ctx(device, outer[0])
            elif not stack:
                stacks.pop(device, None)

    @contextmanager
    def _use_thread_mem_pool(self, device: int, pool, *, dispose_on_exit: bool = False):
        # PyTorch routes allocations to the first matching captures_underway_ entry on the thread,
        # so suspend any outer TMS scope on this (thread, device) while keeping its MemPool alive.
        with self._suspended_outer_pool(device) as stack:
            entry = [pool, self._enter_pool_ctx(device, pool), self]
            pool = None
            stack.append(entry)
            try:
                yield
            finally:
                popped = stack.pop()
                try:
                    popped[1].__exit__(*sys.exc_info())
                finally:
                    if dispose_on_exit:
                        popped.clear()

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
        if getattr(_THREAD_STATE, "active_write_once_depth", 0) > 0:
            raise RuntimeError(
                "Cannot capture a cuda_graph() inside an active write_once=True region; "
                "exit the write_once region first."
            )
        assert self._hook_mode == "preload", "Only hook_mode=preload supports pauseable CUDA Graph currently"
        if is_mlu():
            _sanity_checks()
        with _TorchMemorySaverImpl._region_pause_lock:
            _TorchMemorySaverImpl._active_scopes += 1
        try:
            with self._device_module.graph(
                cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode
            ):
                with self._with_region_config(
                    tag=tag,
                    enable_cpu_backup=enable_cpu_backup,
                    cpu_backup_backend=cpu_backup_backend,
                ):
                    yield
        finally:
            with _TorchMemorySaverImpl._region_pause_lock:
                _TorchMemorySaverImpl._active_scopes -= 1

    @contextmanager
    def _with_region_config(
        self,
        tag: str,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
        enable_disk_backup: bool = False,
        write_once: bool = False,
    ):
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
        with _TorchMemorySaverImpl._region_pause_lock:
            _TorchMemorySaverImpl._active_scopes += 1
        try:
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
                    dispose_on_exit=True,
                ):
                    yield
            finally:
                _THREAD_STATE.active_disable_depth = prev_disable
                self._binary_wrapper.cdll.tms_set_interesting_region(True)
        finally:
            with _TorchMemorySaverImpl._region_pause_lock:
                _TorchMemorySaverImpl._active_scopes -= 1

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

    def _matching_write_once_pools(self, tag: Optional[str]) -> list[tuple[object, tuple]]:
        my_binary = self._binary_wrapper._resolved_binary_path
        matches = []
        for inst in self._instances:
            if inst._binary_wrapper._resolved_binary_path == my_binary:
                for key in inst._write_once_pools:
                    if not tag or key[0] == tag:
                        matches.append((inst, key))
        return matches

    def pause(self, tag: Optional[str]):
        if tag is not None:
            _validate_tag(tag)
        with _TorchMemorySaverImpl._region_pause_lock:
            matching_keys = self._matching_write_once_pools(tag)
            if matching_keys and _TorchMemorySaverImpl._active_scopes > 0:
                raise RuntimeError(
                    "Cannot pause() a write_once region while another region(), disable(), "
                    "or cuda_graph() scope is active; exit active scopes first."
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
            for inst, key in matching_keys:
                inst._write_once_pools.pop(key, None)

    def resume(self, tag: Optional[str]):
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
