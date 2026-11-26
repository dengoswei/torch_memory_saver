import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest


def _run_cmd(cmd, *, env=None, cwd=None, check=True):
    return subprocess.run(
        cmd,
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120.0,
        check=check,
    )


def test_native_direct_io(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "direct.cpp"
    source.write_text(r'''
#include "csrc/disk_backend.cpp"
#include <cassert>

static int write_fault = 0, read_fault = 0;
static size_t short_alignment = 0;

struct TestFile {
    int fd;
    explicit TestFile(int value) : fd(value) {}
    ~TestFile() { if (fd >= 0) close(fd); }
    void finish() { int value = fd; fd = -1; system_check(close(value) == 0, "close"); }
};

extern "C" ssize_t __real_pwrite(int, const void*, size_t, off_t);
extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" ssize_t __wrap_pwrite(int fd, const void* p, size_t n, off_t o) {
    if (write_fault == 1) { write_fault = 2; errno = EINTR; return -1; }
    if (write_fault == 2) { write_fault = 0; n = short_alignment; }
    return __real_pwrite(fd, p, n, o);
}
extern "C" ssize_t __wrap_pread(int fd, void* p, size_t n, off_t o) {
    if (read_fault == 1) { read_fault = 2; errno = EINTR; return -1; }
    if (read_fault == 2) { read_fault = 0; n = short_alignment; }
    return __real_pread(fd, p, n, o);
}

int main(int, char** argv) {
    assert(compute_disk_backup_dir_from_env().empty());
    char name[4096];
    snprintf(name, sizeof(name), "%s/native_XXXXXX", argv[1]);
    TestFile file(mkostemp(name, O_DIRECT | O_CLOEXEC));
    system_check(file.fd >= 0, "create direct fixture");
    uint32_t dio_mask = 0, mem_align = 0, offset_align = 0;
    system_check(query_fd_dioalign(file.fd, dio_mask, mem_align, offset_align), "statx");
    if (!(dio_mask & kTmsStatxDioAlign)) {
        // Test fixture fallback when local FS lacks STATX_DIOALIGN; configure() still requires it.
        mem_align = offset_align = 4096;
    }
    assert(fcntl(file.fd, F_GETFL) & O_DIRECT);
    assert(fcntl(file.fd, F_GETFD) & FD_CLOEXEC);
    size_t align = std::lcm(mem_align, offset_align);
    void* buffer = nullptr;
    assert(posix_memalign(&buffer, std::max(align, sizeof(void*)), 4 * align) == 0);
    for (size_t logical : {size_t(1), align - 1, align, align + 1, 3 * align - 1}) {
        size_t padded = round_up(logical, offset_align);
        for (int cycle = 0; cycle < 3; ++cycle) {
            memset(buffer, 0, padded);
            memset(buffer, cycle + 51, logical);
            assert(ftruncate(file.fd, padded) == 0);
            direct_io(file.fd, buffer, padded, 0, offset_align, mem_align, true);
            memset(buffer, 0xff, padded);
            direct_io(file.fd, buffer, padded, 0, offset_align, mem_align, false);
            for (size_t i = 0; i < padded; ++i)
                assert(static_cast<unsigned char*>(buffer)[i] == (i < logical ? cycle + 51 : 0));
        }
    }
    short_alignment = align;
    memset(buffer, 0x73, 4 * align);
    write_fault = 1;
    direct_io(file.fd, buffer, 4 * align, 0, offset_align, mem_align, true);
    memset(buffer, 0, 4 * align);
    read_fault = 1;
    direct_io(file.fd, buffer, 4 * align, 0, offset_align, mem_align, false);
    assert(write_fault == 0 && read_fault == 0);
    for (size_t i = 0; i < 4 * align; ++i) assert(static_cast<unsigned char*>(buffer)[i] == 0x73);

    bool failed = false;
    assert(ftruncate(file.fd, 0) == 0);
    try { direct_io(file.fd, buffer, align, 0, offset_align, mem_align, false); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed);

    failed = false;
    try { direct_io(file.fd, static_cast<char*>(buffer) + 1, align, 0, offset_align, mem_align, true); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed);

    assert(json_string("a\x01\"\\") == "\"a\\u0001\\\"\\\\\"");
    free(buffer);
    file.finish();
    assert(unlink(name) == 0);
    return 0;
}
''', encoding="utf-8")
    binary = tmp_path / "direct"
    _run_cmd(
        ["g++", "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror", "-DUSE_XPU",
         "-I", str(root), str(source), "-Wl,--wrap=pwrite", "-Wl,--wrap=pread", "-o", str(binary)]
    )
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMS_TEST_DISK_DIR")) as disk_dir:
        env = {k: v for k, v in os.environ.items() if k != "TMS_DISK_BACKUP_DIR"}
        _run_cmd([str(binary), disk_dir], env=env)


def test_region_backup_validation_and_version_gate(tmp_path, monkeypatch):
    import torch
    from torch_memory_saver.binary_wrapper import _setup_function_signatures
    from torch_memory_saver.entrypoint import (
        TorchMemorySaver,
        _resolve_region_backup,
        _torch_supports_detached_mempool_free,
    )

    assert _resolve_region_backup(
        backup="cpu_mmap", write_once=True,
        enable_cpu_backup=None, enable_disk_backup=None,
    ) == (True, False, "mmap")
    assert _resolve_region_backup(
        backup=None, write_once=True,
        enable_cpu_backup=None, enable_disk_backup=True,
    ) == (False, True, None)

    tms = TorchMemorySaver()
    with pytest.raises(TypeError):
        with tms.region(readonly_disk_backup=True):
            pass
    for bad_kwargs in (
        dict(backup="invalid"),
        dict(backup="cpu_mmap", cpu_backup_backend="pinned"),
        dict(backup="cpu", enable_cpu_backup=True),
        dict(write_once=True),
        dict(backup="none", write_once=True),
        dict(backup="cpu", write_once="false"),
        dict(tag="bad\x00tag"),
    ):
        with pytest.raises(ValueError):
            with tms.region(**bad_kwargs):
                pass

    stale_root = tmp_path / "stale_spill"
    for name in ("pid_999999", "pid_123_abc123"):
        stale_sub = stale_root / name
        stale_sub.mkdir(parents=True)
        (stale_sub / "slot_1.bin").write_bytes(b"stale")
    keep_file = stale_root / "keep.txt"
    keep_file.write_bytes(b"keep")
    tms.clear_disk_backup_dir(str(stale_root))
    assert [p.name for p in stale_root.iterdir()] == ["keep.txt"]

    tms_torch = TorchMemorySaver()
    tms_torch.hook_mode = "torch"
    with pytest.raises(ValueError):
        tms_torch.pause("bad\x00tag")
    dangling = stale_root / "dangling"
    dangling.symlink_to(stale_root / "missing_target")
    loop_a = stale_root / "loop_a"
    loop_b = stale_root / "loop_b"
    loop_a.symlink_to(loop_b)
    loop_b.symlink_to(loop_a)
    for bad_dir in (keep_file, keep_file / "child", dangling, dangling / "child", loop_a, loop_a / "child"):
        with pytest.raises(RuntimeError, match="directory"):
            tms_torch.set_disk_backup_dir(str(bad_dir))
    lazy_missing = stale_root / "lazy_missing" / "nested"
    tms_torch.set_disk_backup_dir(str(lazy_missing))
    assert not (stale_root / "lazy_missing").exists()

    for ver in (
        "2.6.0+cu124", "2.7.1", "2.8.0a0+5228986c39", "2.8.0b1",
        "2.8.0rc1", "2.8.0-rc1", "2.8.0preview1", "2.8.0.dev20250201",
        "2.8.0+!", "2.8.0+cu128!", "2.8.0.post1garbage", "invalid-ver",
    ):
        monkeypatch.setattr(torch, "__version__", ver)
        assert not _torch_supports_detached_mempool_free(), ver
    for ver in (
        "2.8", "2.8.0", "2.8.0+cu128", "2.8.0.post1", "2.8.0post1",
        "2.8.0.post1+cu128", "2.8.1a0+git1234", "1!2.0.0",
    ):
        monkeypatch.setattr(torch, "__version__", ver)
        assert _torch_supports_detached_mempool_free(), ver

    req_symbols = (
        "tms_set_current_tag", "tms_get_current_tag",
        "tms_set_interesting_region", "tms_get_interesting_region",
        "tms_set_enable_cpu_backup", "tms_get_enable_cpu_backup",
        "tms_set_cpu_backup_backend", "tms_get_cpu_backup_backend",
        "tms_set_enable_disk_backup", "tms_get_enable_disk_backup",
        "tms_set_write_once", "tms_get_write_once", "tms_supports_write_once",
        "tms_set_disk_backup_dir", "tms_set_disk_backup_dir_checked",
        "tms_configure_direct_disk_backup", "tms_get_disk_backup_info",
        "tms_pause", "tms_resume", "tms_affected_devices",
        "set_memory_margin_bytes", "tms_set_retain_cpu_backup",
        "tms_get_retain_cpu_backup", "tms_get_cpu_backup_pointer",
    )
    with pytest.raises(AttributeError):
        _setup_function_signatures(
            SimpleNamespace(**{name: SimpleNamespace() for name in req_symbols if name != "tms_set_write_once"})
        )


def test_extension_symbol_visibility():
    import torch_memory_saver

    nm_bin = shutil.which("nm")
    if nm_bin is None:
        pytest.skip("nm is not installed")
    root = Path(__file__).resolve().parents[1]
    pkg_dir = Path(torch_memory_saver.__file__).resolve().parent
    binaries = sorted(
        {p.resolve() for search in (root, pkg_dir, pkg_dir.parent) if search.is_dir()
         for p in search.glob("torch_memory_saver_hook_mode_*.so") if p.is_file()}
    )
    if not binaries:
        pytest.skip("no built torch_memory_saver_hook_mode_*.so binaries found")

    for binary in binaries:
        undef = _run_cmd([nm_bin, "-D", "-u", str(binary)]).stdout.splitlines()
        defined = _run_cmd([nm_bin, "-D", "--defined-only", str(binary)]).stdout.splitlines()
        assert not any("statx" in ln for ln in undef), binary.name
        assert not any("DiskBackend" in ln or "TorchMemorySaver" in ln for ln in defined), binary.name
