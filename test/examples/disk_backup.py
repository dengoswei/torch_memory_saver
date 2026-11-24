import ctypes
import fcntl
import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

import torch

from torch_memory_saver import TorchMemorySaver, torch_memory_saver
from torch_memory_saver.entrypoint import _torch_supports_detached_mempool_free
from torch_memory_saver.testing_utils import empty_cache, get_device, is_mlu


def _find_loaded_cudart_path() -> Optional[str]:
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts and "libcudart.so" in parts[-1]:
                    return parts[-1]
    except OSError:
        return None
    return None


def _expected_numa_for_device(device: int) -> int:
    cuda_ver = getattr(torch.version, "cuda", None) or "12"
    cuda_major = cuda_ver.split(".", 1)[0]
    candidates = [None]
    mapped = _find_loaded_cudart_path()
    if mapped:
        candidates.append(mapped)
    candidates.extend([f"libcudart.so.{cuda_major}", "libcudart.so"])

    cudart = None
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
        except OSError:
            continue
        if hasattr(lib, "cudaDeviceGetPCIBusId"):
            cudart = lib
            break
    assert cudart is not None, f"libcudart with cudaDeviceGetPCIBusId not found via {candidates}"
    cudart.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    cudart.cudaDeviceGetPCIBusId.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(64)
    assert cudart.cudaDeviceGetPCIBusId(buf, len(buf), device) == 0
    bdf = buf.value.decode("ascii").lower()
    if len(bdf) > 12 and bdf.startswith("0000") and bdf[8] == ":":
        bdf = bdf[4:]
    numa_path = f"/sys/bus/pci/devices/{bdf}/numa_node"
    assert os.path.exists(numa_path), f"NUMA sysfs entry not found at {numa_path}"
    with open(numa_path, encoding="utf-8") as f:
        node = int(f.read().strip())
    return 0 if node < 0 else node


def run(hook_mode: str):
    # Small chunk (before init) so the tensors below span many chunks.
    os.environ["TMS_DISK_BACKUP_CHUNK_MB"] = "1"
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    device = get_device()

    disk_dir = tempfile.mkdtemp(prefix="tms_disk_backup_", dir=os.environ.get("TMS_TEST_DISK_DIR"))
    print(f"disk_backup_dir={disk_dir}")
    node_env = os.environ.get("TMS_TEST_NUMA_NODE")
    node = int(node_env) if node_env is not None else _expected_numa_for_device(torch.cuda.current_device())
    torch_memory_saver.configure_direct_disk_backup(disk_dir, numa_node=node)
    torch_memory_saver.configure_direct_disk_backup(disk_dir, numa_node=node)
    try:
        torch_memory_saver.configure_direct_disk_backup(disk_dir, numa_node=node + 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("conflicting process-global placement accepted")
    info = torch_memory_saver.get_disk_backup_info()
    assert info["direct"] and info["observed_numa_nodes"] == [node]
    assert info["staging_address"] % info["memory_alignment"] == 0
    assert info["staging_bytes"] == 1024 * 1024
    assert info["fd_flags"] & os.O_DIRECT
    assert info["fd_descriptor_flags"] & fcntl.FD_CLOEXEC
    assert info["active_slots"] == 0 and info["active_padded_bytes"] == 0
    print(f"native_disk_backup_info={info}")
    try:
        _run(disk_dir, device)
    finally:
        shutil.rmtree(disk_dir, ignore_errors=True)


def _run(disk_dir: str, device: str):
    chunk_bytes = 1 * 1024 * 1024
    numel = chunk_bytes * 13 // 4 + 12345  # ~13.4 fp32 chunks: multi-chunk, non-even tail

    print("Allocate tensor_with_disk_backup")
    with torch_memory_saver.region(enable_disk_backup=True):
        big = torch.randn((numel,), dtype=torch.float32, device=device)
        small = torch.full((7,), 10, dtype=torch.uint8, device=device)
    big_oracle = big.detach().to('cpu').clone()
    small_oracle = small.detach().to('cpu').clone()

    print("Allocate tensor_without_backup")
    with torch_memory_saver.region(enable_disk_backup=False):
        no_backup = torch.full((20_000_000,), 20, dtype=torch.uint8, device=device)

    assert torch.equal(big.cpu(), big_oracle)
    assert no_backup[:3].tolist() == [20, 20, 20]

    slot_padded_sizes = set()
    initial = torch_memory_saver.get_disk_backup_info()
    logical_total = None
    for _ in range(3):
        previous = torch_memory_saver.get_disk_backup_info()
        torch_memory_saver.pause()

        # Named per-process slot files exist on disk only while paused.
        info = torch_memory_saver.get_disk_backup_info()
        proc_dir = info["process_directory"]
        assert os.path.isdir(proc_dir)
        assert len(os.listdir(proc_dir)) == info["active_slots"] >= 2, (
            f"expected slot files in {proc_dir}, found {os.listdir(proc_dir)} vs {info}"
        )
        assert info["active_padded_bytes"] >= big.numel() * 4, (
            f"active_padded_bytes smaller than paused tensor: {info}"
        )
        slot_padded_sizes.add(info["active_padded_bytes"])
        # MemPool allocations can share tensors and contain allocator slack.
        actual_logical = info["logical_write_bytes"] - previous["logical_write_bytes"]
        assert actual_logical >= big.numel() * 4 + small.numel()
        if logical_total is None:
            logical_total = actual_logical
        assert actual_logical == logical_total
        assert info["fd_flags"] & os.O_DIRECT
        assert info["fd_descriptor_flags"] & fcntl.FD_CLOEXEC
        assert info["padded_write_bytes"] >= info["logical_write_bytes"]

        occupy = torch.full((20_000_000,), 30, dtype=torch.uint8, device=device)
        del occupy
        empty_cache()

        torch_memory_saver.resume()

        assert torch.equal(big.cpu(), big_oracle)
        assert torch.equal(small.cpu(), small_oracle)
        after_resume = torch_memory_saver.get_disk_backup_info()
        assert after_resume["active_slots"] == 0 and after_resume["active_padded_bytes"] == 0, after_resume
        assert os.listdir(after_resume["process_directory"]) == []
        if is_mlu():
            # CNDriver may reuse the same physical pages. Unbacked contents
            # are unspecified; verify the restored allocation is writable.
            no_backup.fill_(37)
            assert torch.all(no_backup == 37).item()
        else:
            assert no_backup[:3].tolist() != [20, 20, 20]

    active_padded = next(iter(slot_padded_sizes))
    info = torch_memory_saver.get_disk_backup_info()
    assert info["logical_write_bytes"] - initial["logical_write_bytes"] == 3 * logical_total
    assert info["logical_read_bytes"] - initial["logical_read_bytes"] == 3 * logical_total
    assert info["padded_write_bytes"] - initial["padded_write_bytes"] == 3 * active_padded
    assert info["padded_read_bytes"] - initial["padded_read_bytes"] == 3 * active_padded
    assert len(slot_padded_sizes) == 1, f"disk usage grew across cycles: {sorted(slot_padded_sizes)}"
    print(f"OK: {big.numel()} fp32 elems restored bit-identically across 3 cycles, "
          f"active_padded constant at {active_padded} bytes")

    # Nested mutable region and disable() must not reuse an outer backup="none" pool block.
    with torch_memory_saver.region(tag="nest_outer_none", backup="none"):
        discarded = torch.full((4096,), 5.0, dtype=torch.float32, device=device)
        del discarded
        with torch_memory_saver.region(tag="nest_inner_disk", backup="disk"):
            inner_disk_t = torch.full((4096,), 31.0, dtype=torch.float32, device=device)
        with torch_memory_saver.disable():
            disabled_t = torch.full((4096,), 77.0, dtype=torch.float32, device=device)
        outer_live_t = torch.full((4096,), 19.0, dtype=torch.float32, device=device)

    bytes_before_nested_pause = torch_memory_saver.get_disk_backup_info()["logical_write_bytes"]
    torch_memory_saver.pause("nest_inner_disk")
    bytes_after_nested_pause = torch_memory_saver.get_disk_backup_info()["logical_write_bytes"]
    assert bytes_after_nested_pause > bytes_before_nested_pause, (
        bytes_before_nested_pause,
        bytes_after_nested_pause,
    )
    assert outer_live_t.cpu().eq(19.0).all().item()
    assert disabled_t.cpu().eq(77.0).all().item()
    torch_memory_saver.resume("nest_inner_disk")
    assert inner_disk_t.cpu().eq(31.0).all().item()

    torch_memory_saver.pause()
    torch_memory_saver.resume()
    assert inner_disk_t.cpu().eq(31.0).all().item()
    assert disabled_t.cpu().eq(77.0).all().item()
    del inner_disk_t, disabled_t, outer_live_t
    empty_cache()


def run_auto_numa_and_multi_instance(hook_mode: str):
    os.environ["TMS_DISK_BACKUP_CHUNK_MB"] = "1"
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    expected_numa = _expected_numa_for_device(torch.cuda.current_device())
    with (
        tempfile.TemporaryDirectory(prefix="tms_disk_a_", dir=os.environ.get("TMS_TEST_DISK_DIR")) as disk_dir,
        tempfile.TemporaryDirectory(prefix="tms_disk_b_", dir=os.environ.get("TMS_TEST_DISK_DIR")) as other_dir,
    ):
        torch_memory_saver.configure_direct_disk_backup(disk_dir)
        info1 = torch_memory_saver.get_disk_backup_info()
        assert info1["direct"] is True
        assert info1["requested_numa_node"] == expected_numa
        assert info1["observed_numa_nodes"] == [expected_numa]

        # A second TorchMemorySaver() shares the process-global DiskBackend idempotently.
        tms2 = TorchMemorySaver()
        tms2.hook_mode = hook_mode
        tms2.configure_direct_disk_backup(disk_dir + "/")
        tms2.configure_direct_disk_backup(disk_dir, numa_node=expected_numa)
        tms2.set_disk_backup_dir(disk_dir)
        info2 = tms2.get_disk_backup_info()
        assert info2["staging_address"] == info1["staging_address"]

        nonexistent_cfg = os.path.join(disk_dir, "never_created_cfg")
        nonexistent_dir = os.path.join(disk_dir, "never_created_dir")
        for bad_call in (
            lambda: tms2.configure_direct_disk_backup(other_dir),
            lambda: tms2.configure_direct_disk_backup(nonexistent_cfg),
            lambda: tms2.configure_direct_disk_backup(disk_dir, numa_node=expected_numa + 1),
            lambda: tms2.set_disk_backup_dir(other_dir),
            lambda: tms2.set_disk_backup_dir(nonexistent_dir),
        ):
            try:
                bad_call()
            except RuntimeError:
                pass
            else:
                raise AssertionError("conflicting process-global placement accepted")
        assert not os.path.exists(nonexistent_cfg)
        assert not os.path.exists(nonexistent_dir)

        ro_new_oracle = torch.randn((512, 1024), dtype=torch.float32, device="cuda").cpu()
        ro_legacy_oracle = torch.randn((256, 1024), dtype=torch.float32, device="cuda").cpu()
        rw_initial = torch.full((256, 1024), 3.0, dtype=torch.float32)
        rw_updated = torch.full((256, 1024), 7.0, dtype=torch.float32)

        write_once_supported = not (
            hook_mode == "torch" and not _torch_supports_detached_mempool_free()
        )
        if not write_once_supported:
            try:
                with torch_memory_saver.region(tag="ro_new", backup="disk", write_once=True):
                    pass
            except NotImplementedError:
                pass
            else:
                raise AssertionError("expected NotImplementedError for write_once in torch mode < 2.8")

            with tms2.region(tag="rw", backup="disk"):
                rw = rw_initial.to("cuda")
            torch_memory_saver.pause()
            torch_memory_saver.resume()
            assert torch.equal(rw.cpu(), rw_initial)
            rw.copy_(rw_updated.to("cuda"))
            torch_memory_saver.pause()
            torch_memory_saver.resume()
            assert torch.equal(rw.cpu(), rw_updated)
            del rw
            gc.collect()
            torch.cuda.empty_cache()
            return

        with torch_memory_saver.region(tag="ro_open", backup="disk", write_once=True):
            _ = torch.empty((1024,), dtype=torch.float32, device="cuda")
            for bad_pause in (
                lambda: torch_memory_saver.pause(),
                lambda: torch_memory_saver.pause("ro_open"),
                lambda: tms2.pause(),
            ):
                try:
                    bad_pause()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("pause() succeeded inside an open write_once region")
        torch.cuda.empty_cache()

        with torch_memory_saver.region(tag="ro_new", backup="disk", write_once=True):
            ro_new = ro_new_oracle.to("cuda")
        with tms2.region(tag="ro_legacy", enable_disk_backup=True, readonly_disk_backup=True):
            ro_legacy = ro_legacy_oracle.to("cuda")
        with tms2.region(tag="rw", backup="disk"):
            rw = rw_initial.to("cuda")

        torch_memory_saver.pause()
        after_pause1 = tms2.get_disk_backup_info()
        assert len(os.listdir(after_pause1["process_directory"])) == after_pause1["active_slots"] >= 3
        assert after_pause1["readonly_skipped_pauses"] == 0
        pause1_written = after_pause1["logical_write_bytes"] - info2["logical_write_bytes"]

        torch_memory_saver.resume()
        assert torch.equal(ro_new.cpu(), ro_new_oracle)
        assert torch.equal(ro_legacy.cpu(), ro_legacy_oracle)
        assert torch.equal(rw.cpu(), rw_initial)

        # Mutate all tensors on GPU: write-once slots must ignore GPU edits on 2nd pause.
        ro_new.fill_(999.0)
        ro_legacy.fill_(888.0)
        rw.copy_(rw_updated.to("cuda"))

        torch_memory_saver.pause()
        after_pause2 = torch_memory_saver.get_disk_backup_info()
        assert len(os.listdir(after_pause2["process_directory"])) == after_pause2["active_slots"] >= 2
        assert after_pause2["readonly_skipped_pauses"] >= 2
        assert after_pause2["readonly_skipped_write_bytes"] >= (
            ro_new.numel() + ro_legacy.numel()
        ) * 4
        pause2_written = after_pause2["logical_write_bytes"] - after_pause1["logical_write_bytes"]
        assert pause2_written < pause1_written

        torch_memory_saver.resume()
        assert torch.equal(ro_new.cpu(), ro_new_oracle)
        assert torch.equal(ro_legacy.cpu(), ro_legacy_oracle)
        assert torch.equal(rw.cpu(), rw_updated)

        # Same-tag write_once replacement must not inherit the freed tensor's cached block.
        ro_a_oracle = torch.full((256, 1024), 11.0, dtype=torch.float32)
        ro_b_oracle = torch.full((256, 1024), 23.0, dtype=torch.float32)
        with torch_memory_saver.region(tag="ro_disk_reuse", backup="disk", write_once=True):
            ro_a = ro_a_oracle.to("cuda")
        torch_memory_saver.pause("ro_disk_reuse")
        torch_memory_saver.resume("ro_disk_reuse")
        assert torch.equal(ro_a.cpu(), ro_a_oracle)
        del ro_a
        with torch_memory_saver.region(tag="ro_disk_reuse", backup="disk", write_once=True):
            ro_b = ro_b_oracle.to("cuda")
        torch_memory_saver.pause("ro_disk_reuse")
        torch_memory_saver.resume("ro_disk_reuse")
        assert torch.equal(ro_b.cpu(), ro_b_oracle)

        # Cross-instance pause() retires pools owned by sibling TorchMemorySaver instances.
        xi_a_oracle = torch.full((256, 1024), 31.0, dtype=torch.float32)
        xi_b_oracle = torch.full((256, 1024), 47.0, dtype=torch.float32)
        with tms2.region(tag="ro_disk_xinst", backup="disk", write_once=True):
            xi_a = xi_a_oracle.to("cuda")
        torch_memory_saver.pause("ro_disk_xinst")
        torch_memory_saver.resume("ro_disk_xinst")
        assert torch.equal(xi_a.cpu(), xi_a_oracle)
        del xi_a
        with tms2.region(tag="ro_disk_xinst", backup="disk", write_once=True):
            xi_b = xi_b_oracle.to("cuda")
        torch_memory_saver.pause("ro_disk_xinst")
        torch_memory_saver.resume("ro_disk_xinst")
        assert torch.equal(xi_b.cpu(), xi_b_oracle)
        del xi_b
        gc.collect()
        torch.cuda.empty_cache()

        # Retiring a write_once pool via pause() while an unrelated mutable region is active on the device.
        ro_under_mut_oracle = torch.full((256, 1024), 53.0, dtype=torch.float32)
        mut_hold_oracle = torch.full((256, 1024), 61.0, dtype=torch.float32)
        with torch_memory_saver.region(tag="ro_pause_under_mut", backup="disk", write_once=True):
            ro_under_mut = ro_under_mut_oracle.to("cuda")
        with torch_memory_saver.region(tag="mut_active_during_ro_pause", backup="disk"):
            mut_live = mut_hold_oracle.to("cuda")
            torch_memory_saver.pause("ro_pause_under_mut")
            torch_memory_saver.resume("ro_pause_under_mut")
        assert torch.equal(ro_under_mut.cpu(), ro_under_mut_oracle)
        assert torch.equal(mut_live.cpu(), mut_hold_oracle)
        del ro_under_mut, mut_live
        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.device_count() >= 2:
            cross_oracle = torch.full((64, 64), 97.0, dtype=torch.float32)
            with torch.cuda.device(1):
                with torch_memory_saver.region(tag="ro_cross_dev", backup="disk", write_once=True):
                    cross_ro = cross_oracle.to("cuda:1")
            assert torch.cuda.current_device() == 0
            torch_memory_saver.pause("ro_cross_dev")
            assert torch.cuda.current_device() == 0
            torch_memory_saver.resume("ro_cross_dev")
            assert torch.cuda.current_device() == 0
            assert torch.equal(cross_ro.cpu(), cross_oracle)
            del cross_ro
            gc.collect()
            torch.cuda.empty_cache()

        # Repeated retirement + empty_cache() returns disk slots and bytes to baseline.
        baseline = torch_memory_saver.get_disk_backup_info()
        for cycle in range(3):
            cycle_tag = f"ro_disk_reclaim_{cycle}"
            cycle_oracle = torch.randn((256, 1024), dtype=torch.float32)
            with torch_memory_saver.region(tag=cycle_tag, backup="disk", write_once=True):
                cycle_t = cycle_oracle.to("cuda")
            torch_memory_saver.pause(tag=cycle_tag)
            mid = torch_memory_saver.get_disk_backup_info()
            assert mid["active_slots"] > baseline["active_slots"]
            assert mid["active_padded_bytes"] > baseline["active_padded_bytes"]
            torch_memory_saver.resume(tag=cycle_tag)
            assert torch.equal(cycle_t.cpu(), cycle_oracle)
            del cycle_t
            gc.collect()
            torch.cuda.empty_cache()
            after_gc = torch_memory_saver.get_disk_backup_info()
            assert after_gc["active_slots"] == baseline["active_slots"]
            assert after_gc["active_padded_bytes"] == baseline["active_padded_bytes"]


def run_env_and_set_dir_auto_configure(hook_mode: str):
    env_base = os.environ.copy()
    env_base["TMS_DISK_BACKUP_CHUNK_MB"] = "1"
    env_base.pop("TMS_DISK_BACKUP_DIR", None)

    def _run_child(script: str, extra_env=None, check: bool = True):
        env = env_base.copy()
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
        )
        if check:
            assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc

    env_write_once = not (
        hook_mode == "torch" and not _torch_supports_detached_mempool_free()
    )
    with tempfile.TemporaryDirectory(prefix="tms_env_dir_", dir=os.environ.get("TMS_TEST_DISK_DIR")) as d_env:
        _run_child(
            f"""
import gc, os, torch
from torch_memory_saver import torch_memory_saver
torch_memory_saver.hook_mode = {hook_mode!r}
oracle = torch.arange(1024, dtype=torch.float32)
with torch_memory_saver.region(tag="env_ro", backup="disk", write_once={env_write_once!r}):
    t = oracle.to("cuda")
info = torch_memory_saver.get_disk_backup_info()
assert info["direct"] is True
torch_memory_saver.pause()
pause_info = torch_memory_saver.get_disk_backup_info()
assert len(os.listdir(pause_info["process_directory"])) == pause_info["active_slots"] >= 1
torch_memory_saver.resume()
assert torch.equal(t.cpu(), oracle)
if {env_write_once!r}:
    t.fill_(-5.0)
    torch_memory_saver.pause()
    torch_memory_saver.resume()
    assert torch.equal(t.cpu(), oracle)
    assert torch_memory_saver.get_disk_backup_info()["readonly_skipped_pauses"] >= 1
del t
torch_memory_saver._impl._clear_all_pools()
gc.collect()
torch.cuda.empty_cache()
final_info = torch_memory_saver.get_disk_backup_info()
assert final_info["active_slots"] == 0
assert final_info["active_padded_bytes"] == 0
""",
            extra_env={"TMS_DISK_BACKUP_DIR": d_env},
        )
        assert os.listdir(d_env) == []

    with tempfile.TemporaryDirectory(prefix="tms_set_dir_", dir=os.environ.get("TMS_TEST_DISK_DIR")) as d_set:
        _run_child(
            f"""
import gc, os, torch
from torch_memory_saver import torch_memory_saver
torch_memory_saver.hook_mode = {hook_mode!r}
torch_memory_saver.set_disk_backup_dir({d_set!r})
oracle = torch.full((2048,), 42, dtype=torch.uint8)
with torch_memory_saver.region(tag="setdir", backup="disk"):
    t = oracle.to("cuda")
# Repeating canonical directory after auto-configure is idempotent; mismatch fails without mkdir.
torch_memory_saver.set_disk_backup_dir({d_set!r} + "/")
mismatch_path = os.path.join({d_set!r}, "never_created_mismatch")
try:
    torch_memory_saver.set_disk_backup_dir(mismatch_path)
except RuntimeError:
    pass
else:
    raise AssertionError("set_disk_backup_dir accepted mismatched path after configuration")
assert not os.path.exists(mismatch_path)
info = torch_memory_saver.get_disk_backup_info()
assert info["direct"] is True and info["directory"] == os.path.realpath({d_set!r})
torch_memory_saver.pause()
pause_info = torch_memory_saver.get_disk_backup_info()
assert len(os.listdir(pause_info["process_directory"])) == pause_info["active_slots"] >= 1
torch_memory_saver.resume()
assert torch.equal(t.cpu(), oracle)
assert torch_memory_saver.get_disk_backup_info()["active_slots"] == 0
del t
torch_memory_saver._impl._clear_all_pools()
gc.collect()
torch.cuda.empty_cache()
final_info = torch_memory_saver.get_disk_backup_info()
assert final_info["active_slots"] == 0
assert final_info["active_padded_bytes"] == 0
"""
        )
        assert os.listdir(d_set) == []

    with tempfile.TemporaryDirectory(prefix="tms_kill_dir_", dir=os.environ.get("TMS_TEST_DISK_DIR")) as d_exit:
        _run_child(
            f"""
import os, torch
from torch_memory_saver import torch_memory_saver
torch_memory_saver.hook_mode = {hook_mode!r}
torch_memory_saver.configure_direct_disk_backup({d_exit!r})
with torch_memory_saver.region(tag="hard_exit", backup="disk"):
    _t = torch.full((1024 * 1024,), 9, dtype=torch.uint8, device="cuda")
torch_memory_saver.pause()
pause_info = torch_memory_saver.get_disk_backup_info()
assert pause_info["active_slots"] >= 1
assert len(os.listdir(pause_info["process_directory"])) == pause_info["active_slots"]
os._exit(0)
"""
        )
        assert len(os.listdir(d_exit)) == 1
        torch_memory_saver.clear_disk_backup_dir(d_exit)
        assert os.listdir(d_exit) == []

    # Entering a disk region without configuring a directory fails recoverably in Python.
    proc = _run_child(
        f"""
import torch
from torch_memory_saver import torch_memory_saver
torch_memory_saver.hook_mode = {hook_mode!r}
try:
    with torch_memory_saver.region(backup="disk"):
        _ = torch.empty((1024,), device="cuda")
except RuntimeError as exc:
    assert "configure disk backup first" in str(exc), str(exc)
else:
    raise AssertionError("unconfigured disk region succeeded")
# Process remains usable after catching the recoverable RuntimeError.
assert torch.zeros((4,), device="cuda").sum().item() == 0
""",
    )
    assert proc.returncode == 0

    if hook_mode == "preload":
        # Env-driven preload disk allocation without directory aborts in native malloc.
        proc_unconfigured_env = _run_child(
            """
import torch
from torch_memory_saver import torch_memory_saver
torch_memory_saver.hook_mode = "preload"
_ = torch.empty((1024,), device="cuda")
""",
            extra_env={
                "TMS_INIT_ENABLE": "1",
                "TMS_INIT_ENABLE_DISK_BACKUP": "1",
            },
            check=False,
        )
        assert proc_unconfigured_env.returncode != 0
        assert "configure disk backup first" in (
            proc_unconfigured_env.stdout + proc_unconfigured_env.stderr
        )

        # Reject TMS_INIT_READONLY_DISK_BACKUP=1 across both invalid CPU and unsafe preload disk paths.
        with tempfile.TemporaryDirectory(
            prefix="tms_ro_env_", dir=os.environ.get("TMS_TEST_DISK_DIR")
        ) as d_ro_env:
            for extra_env in (
                {
                    "TMS_INIT_ENABLE": "1",
                    "TMS_INIT_ENABLE_CPU_BACKUP": "1",
                    "TMS_INIT_READONLY_DISK_BACKUP": "1",
                },
                {
                    "TMS_INIT_ENABLE": "1",
                    "TMS_INIT_ENABLE_DISK_BACKUP": "1",
                    "TMS_INIT_READONLY_DISK_BACKUP": "1",
                    "TMS_DISK_BACKUP_DIR": d_ro_env,
                },
            ):
                proc_ro_env = _run_child(
                    """
import torch
from torch_memory_saver import torch_memory_saver
torch_memory_saver.hook_mode = "preload"
_ = torch.empty((1024,), device="cuda")
""",
                    extra_env=extra_env,
                    check=False,
                )
                assert proc_ro_env.returncode != 0
                assert "TMS_INIT_READONLY_DISK_BACKUP is unsupported" in (
                    proc_ro_env.stdout + proc_ro_env.stderr
                )
            assert os.listdir(d_ro_env) == []


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
