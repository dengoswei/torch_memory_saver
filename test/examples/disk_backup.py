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
    os.environ["TMS_DISK_BACKUP_CHUNK_MB"] = "1"
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    device = get_device()

    disk_dir = tempfile.mkdtemp(prefix="tms_disk_backup_", dir=os.environ.get("TMS_TEST_DISK_DIR"))
    node_env = os.environ.get("TMS_TEST_NUMA_NODE")
    node = int(node_env) if node_env is not None else _expected_numa_for_device(torch.cuda.current_device())
    torch_memory_saver.configure_direct_disk_backup(disk_dir, numa_node=node)
    info = torch_memory_saver.get_disk_backup_info()
    assert info["direct"] and info["observed_numa_nodes"] == [node]
    assert info["staging_address"] % info["memory_alignment"] == 0
    assert info["staging_bytes"] == 1024 * 1024
    assert info["fd_flags"] & os.O_DIRECT
    assert info["fd_descriptor_flags"] & fcntl.FD_CLOEXEC
    assert info["active_slots"] == 0 and info["active_padded_bytes"] == 0
    try:
        _run(device)
    finally:
        shutil.rmtree(disk_dir, ignore_errors=True)


def _run(device: str):
    numel = (13 * 1024 * 1024) // 4 + 12345  # multi-chunk with non-aligned tail
    with torch_memory_saver.region(enable_disk_backup=True):
        big = torch.randn((numel,), dtype=torch.float32, device=device)
        small = torch.full((7,), 10, dtype=torch.uint8, device=device)
    big_oracle = big.detach().to("cpu").clone()
    small_oracle = small.detach().to("cpu").clone()

    with torch_memory_saver.region(enable_disk_backup=False):
        no_backup = torch.full((20_000_000,), 20, dtype=torch.uint8, device=device)

    slot_padded_sizes = set()
    initial = torch_memory_saver.get_disk_backup_info()
    logical_total = None
    for _ in range(3):
        previous = torch_memory_saver.get_disk_backup_info()
        torch_memory_saver.pause()

        info = torch_memory_saver.get_disk_backup_info()
        proc_dir = info["process_directory"]
        assert os.path.isdir(proc_dir)
        assert len(os.listdir(proc_dir)) == info["active_slots"] >= 2
        assert info["active_padded_bytes"] >= big.numel() * 4
        slot_padded_sizes.add(info["active_padded_bytes"])
        actual_logical = info["logical_write_bytes"] - previous["logical_write_bytes"]
        assert actual_logical >= big.numel() * 4 + small.numel()
        logical_total = actual_logical if logical_total is None else logical_total
        assert actual_logical == logical_total

        occupy = torch.full((20_000_000,), 30, dtype=torch.uint8, device=device)
        del occupy
        empty_cache()

        torch_memory_saver.resume()
        assert torch.equal(big.cpu(), big_oracle)
        assert torch.equal(small.cpu(), small_oracle)
        after_resume = torch_memory_saver.get_disk_backup_info()
        assert after_resume["active_slots"] == 0 and after_resume["active_padded_bytes"] == 0
        assert os.listdir(after_resume["process_directory"]) == []
        if is_mlu():
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
    assert len(slot_padded_sizes) == 1

    # Nested mutable region and disable() must not reuse an outer backup="none" pool block.
    with torch_memory_saver.region(tag="nest_outer_none", backup="none"):
        discarded = torch.full((4096,), 5.0, dtype=torch.float32, device=device)
        del discarded
        with torch_memory_saver.region(tag="nest_inner_disk", backup="disk"):
            inner_disk_t = torch.full((4096,), 31.0, dtype=torch.float32, device=device)
        with torch_memory_saver.disable():
            disabled_t = torch.full((4096,), 77.0, dtype=torch.float32, device=device)
        outer_live_t = torch.full((4096,), 19.0, dtype=torch.float32, device=device)

    before_nested = torch_memory_saver.get_disk_backup_info()["logical_write_bytes"]
    torch_memory_saver.pause("nest_inner_disk")
    assert torch_memory_saver.get_disk_backup_info()["logical_write_bytes"] > before_nested
    assert outer_live_t.cpu().eq(19.0).all().item()
    assert disabled_t.cpu().eq(77.0).all().item()
    torch_memory_saver.resume("nest_inner_disk")
    assert inner_disk_t.cpu().eq(31.0).all().item()
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
        assert info1["direct"] is True and info1["observed_numa_nodes"] == [expected_numa]

        tms2 = TorchMemorySaver()
        tms2.hook_mode = hook_mode
        tms2.configure_direct_disk_backup(disk_dir + "/")
        assert tms2.get_disk_backup_info()["staging_address"] == info1["staging_address"]

        nonexistent = os.path.join(disk_dir, "never_created")
        for bad_call in (
            lambda: tms2.configure_direct_disk_backup(other_dir),
            lambda: tms2.configure_direct_disk_backup(disk_dir, numa_node=expected_numa + 1),
            lambda: tms2.set_disk_backup_dir(nonexistent),
        ):
            try:
                bad_call()
            except RuntimeError:
                pass
            else:
                raise AssertionError("conflicting process-global placement accepted")
        assert not os.path.exists(nonexistent)

        if hook_mode == "torch" and not _torch_supports_detached_mempool_free():
            try:
                with torch_memory_saver.region(tag="ro_unsupported", backup="disk", write_once=True):
                    pass
            except NotImplementedError:
                return
            raise AssertionError("expected NotImplementedError for write_once in torch mode < 2.8")

        with torch_memory_saver.region(tag="ro_open", backup="disk", write_once=True):
            open_ro = torch.empty((1024,), dtype=torch.float32, device="cuda")
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
        with torch_memory_saver.region(tag="mut_scope", backup="disk"):
            try:
                torch_memory_saver.pause("ro_open")
            except RuntimeError:
                pass
            else:
                raise AssertionError("pause(write_once) succeeded while another scope was active")
        del open_ro
        empty_cache()

        ro_new_oracle = torch.randn((512, 1024), dtype=torch.float32)
        ro_legacy_oracle = torch.randn((256, 1024), dtype=torch.float32)
        rw_initial = torch.full((256, 1024), 3.0, dtype=torch.float32)
        rw_updated = torch.full((256, 1024), 7.0, dtype=torch.float32)

        with torch_memory_saver.region(tag="ro_new", backup="disk", write_once=True):
            ro_new = ro_new_oracle.to("cuda")
        with tms2.region(tag="ro_legacy", enable_disk_backup=True, write_once=True):
            ro_legacy = ro_legacy_oracle.to("cuda")
        with tms2.region(tag="rw", backup="disk"):
            rw = rw_initial.to("cuda")

        torch_memory_saver.pause()
        p1 = tms2.get_disk_backup_info()
        assert len(os.listdir(p1["process_directory"])) == p1["active_slots"] >= 3
        torch_memory_saver.resume()
        assert torch.equal(ro_new.cpu(), ro_new_oracle)
        assert torch.equal(ro_legacy.cpu(), ro_legacy_oracle)
        assert torch.equal(rw.cpu(), rw_initial)

        ro_new.fill_(999.0)
        ro_legacy.fill_(888.0)
        rw.copy_(rw_updated.to("cuda"))
        torch_memory_saver.pause()
        p2 = torch_memory_saver.get_disk_backup_info()
        assert p2["readonly_skipped_pauses"] >= 2
        assert p2["readonly_skipped_write_bytes"] >= (ro_new.numel() + ro_legacy.numel()) * 4
        assert (p2["logical_write_bytes"] - p1["logical_write_bytes"]) < (p1["logical_write_bytes"] - info1["logical_write_bytes"])
        torch_memory_saver.resume()
        assert torch.equal(ro_new.cpu(), ro_new_oracle)
        assert torch.equal(ro_legacy.cpu(), ro_legacy_oracle)
        assert torch.equal(rw.cpu(), rw_updated)
        del ro_new, ro_legacy, rw
        gc.collect()
        empty_cache()

        # Same-tag write_once reuse + slot reclamation on del + empty_cache().
        baseline = torch_memory_saver.get_disk_backup_info()
        for cycle in range(2):
            oracle = torch.full((256, 1024), float(11 + cycle), dtype=torch.float32)
            with tms2.region(tag="ro_reuse", backup="disk", write_once=True):
                t = oracle.to("cuda")
            torch_memory_saver.pause("ro_reuse")
            torch_memory_saver.resume("ro_reuse")
            assert torch.equal(t.cpu(), oracle)
            del t
            gc.collect()
            empty_cache()
            after = torch_memory_saver.get_disk_backup_info()
            assert after["active_slots"] == baseline["active_slots"]
            assert after["active_padded_bytes"] == baseline["active_padded_bytes"]

        if torch.cuda.device_count() >= 2:
            cross_oracle = torch.full((64, 64), 97.0, dtype=torch.float32)
            with torch.cuda.device(1):
                with torch_memory_saver.region(tag="ro_cross_dev", backup="disk", write_once=True):
                    cross_ro = cross_oracle.to("cuda:1")
            torch_memory_saver.pause("ro_cross_dev")
            torch_memory_saver.resume("ro_cross_dev")
            assert torch.cuda.current_device() == 0
            assert torch.equal(cross_ro.cpu(), cross_oracle)


def run_env_and_set_dir_auto_configure(hook_mode: str):
    env_base = os.environ.copy()
    env_base["TMS_DISK_BACKUP_CHUNK_MB"] = "1"
    env_base.pop("TMS_DISK_BACKUP_DIR", None)

    def _run_child(script: str, extra_env=None, check: bool = True):
        env = env_base.copy()
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        if check:
            assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc

    with tempfile.TemporaryDirectory(prefix="tms_env_dir_", dir=os.environ.get("TMS_TEST_DISK_DIR")) as d_env:
        _run_child(
            f"""
import gc, os, torch
from torch_memory_saver import torch_memory_saver
torch_memory_saver.hook_mode = {hook_mode!r}
oracle = torch.arange(1024, dtype=torch.float32)
with torch_memory_saver.region(tag="env", backup="disk"):
    t = oracle.to("cuda")
assert torch_memory_saver.get_disk_backup_info()["direct"] is True
torch_memory_saver.pause()
torch_memory_saver.resume()
assert torch.equal(t.cpu(), oracle)
del t; gc.collect(); torch.cuda.empty_cache()
assert torch_memory_saver.get_disk_backup_info()["active_slots"] == 0
""",
            extra_env={"TMS_DISK_BACKUP_DIR": d_env},
        )
        assert os.listdir(d_env) == []

    with tempfile.TemporaryDirectory(prefix="tms_set_dir_", dir=os.environ.get("TMS_TEST_DISK_DIR")) as d_set:
        _run_child(
            f"""
import os, torch
from torch_memory_saver import torch_memory_saver
torch_memory_saver.hook_mode = {hook_mode!r}
torch_memory_saver.set_disk_backup_dir({d_set!r})
with torch_memory_saver.region(tag="exit", backup="disk"):
    _t = torch.full((4096,), 9, dtype=torch.uint8, device="cuda")
torch_memory_saver.pause()
assert torch_memory_saver.get_disk_backup_info()["active_slots"] >= 1
os._exit(0)
"""
        )
        assert len(os.listdir(d_set)) == 1
        torch_memory_saver.clear_disk_backup_dir(d_set)
        assert os.listdir(d_set) == []

    _run_child(
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
assert torch.zeros((4,), device="cuda").sum().item() == 0
"""
    )


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
