from collections import defaultdict
from contextlib import contextmanager, nullcontext
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import zipfile

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


def test_verify_staging_pages_numa_migration_retry(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "numa_retry.cpp"
    source.write_text(r'''
#include "csrc/disk_backend.cpp"
#include <cassert>

int main() {
    long page = sysconf(_SC_PAGESIZE);
    assert(page > 0);
    size_t page_bytes = static_cast<size_t>(page);
    size_t chunk_bytes = 4 * page_bytes;
    void* buf = mmap(nullptr, chunk_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(buf != MAP_FAILED);
    memset(buf, 0x3a, chunk_bytes);

    int calls = 0;
    verify_staging_pages_numa(buf, chunk_bytes, page_bytes, 0, [&](size_t count, void**, int* st) {
        ++calls;
        for (size_t i = 0; i < count; ++i) st[i] = 0;
        if (calls <= 2) st[2] = -ENOENT;
        return 0;
    });
    assert(calls == 3);

    bool caught_wrong_node = false;
    try {
        verify_staging_pages_numa(buf, chunk_bytes, page_bytes, 0, [&](size_t count, void**, int* st) {
            for (size_t i = 0; i < count; ++i) st[i] = 0;
            st[0] = 1;
            return 0;
        });
    } catch (const std::runtime_error& e) {
        caught_wrong_node = std::string(e.what()).find("observed 1") != std::string::npos;
    }
    assert(caught_wrong_node);
    munmap(buf, chunk_bytes);
    return 0;
}
''', encoding="utf-8")
    binary = tmp_path / "numa_retry"
    _run_cmd(
        ["g++", "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
         "-DUSE_XPU", "-I", str(root), str(source), "-o", str(binary)]
    )
    _run_cmd([str(binary)])


def _write_cuda_shims(shim_dir: Path) -> None:
    (shim_dir / "cuda.h").write_text(r'''
#pragma once
#include <cstdint>
#include <cstdlib>
typedef int CUresult;
typedef int CUdevice;
typedef uint64_t CUdeviceptr;
typedef uint64_t CUmemGenericAllocationHandle;
typedef void* CUcontext;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CU_MEM_ALLOCATION_TYPE_PINNED 1
#define CU_MEM_LOCATION_TYPE_DEVICE 1
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE 3
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM 0
#define CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED 100
struct CUmemLocation { int type; int id; };
struct CUmemAllocFlags { int gpuDirectRDMACapable; };
struct CUmemAllocationProp { int type; CUmemLocation location; CUmemAllocFlags allocFlags; };
struct CUmemAccessDesc { CUmemLocation location; int flags; };
inline CUresult cuGetErrorName(CUresult, const char** p) { *p = "CUDA_SUCCESS"; return CUDA_SUCCESS; }
inline CUresult cuGetErrorString(CUresult, const char** p) { *p = "mock"; return CUDA_SUCCESS; }
inline CUresult cuDeviceGetAttribute(int* flag, int, CUdevice) { *flag = 0; return CUDA_SUCCESS; }
inline CUresult cuMemGetAllocationGranularity(size_t* g, const CUmemAllocationProp*, int) { *g = 4096; return CUDA_SUCCESS; }
inline CUresult cuMemCreate(CUmemGenericAllocationHandle* h, size_t, const CUmemAllocationProp*, unsigned long long) { *h = 1; return CUDA_SUCCESS; }
inline CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t, CUdeviceptr, unsigned long long) {
    void* p = std::calloc(1, size ? size : 1);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    *ptr = reinterpret_cast<CUdeviceptr>(p);
    return CUDA_SUCCESS;
}
inline CUresult cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) { return CUDA_SUCCESS; }
inline CUresult cuMemUnmap(CUdeviceptr, size_t) { return CUDA_SUCCESS; }
inline CUresult cuMemRelease(CUmemGenericAllocationHandle) { return CUDA_SUCCESS; }
inline CUresult cuMemAddressFree(CUdeviceptr ptr, size_t) { std::free(reinterpret_cast<void*>(ptr)); return CUDA_SUCCESS; }
inline CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) { return CUDA_SUCCESS; }
inline int& tms_shim_current_device() { static thread_local int d = 0; return d; }
inline CUcontext& tms_shim_current_ctx() { static thread_local CUcontext c = nullptr; return c; }
inline CUresult cuCtxGetCurrent(CUcontext* pctx) { *pctx = tms_shim_current_ctx(); return CUDA_SUCCESS; }
inline CUresult cuCtxSetCurrent(CUcontext ctx) {
    tms_shim_current_ctx() = ctx;
    if (ctx != nullptr) tms_shim_current_device() = static_cast<int>(reinterpret_cast<uintptr_t>(ctx) & 0xff) - 1;
    return CUDA_SUCCESS;
}
inline CUresult cuCtxGetDevice(CUdevice* d) { *d = tms_shim_current_device(); return CUDA_SUCCESS; }
inline CUresult cuDeviceGet(CUdevice* d, int ord) { *d = ord; return CUDA_SUCCESS; }
''', encoding="utf-8")
    (shim_dir / "cuda_runtime_api.h").write_text(r'''
#pragma once
#include "cuda.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
typedef int cudaError_t;
typedef void* cudaStream_t;
enum cudaMemcpyKind { cudaMemcpyDeviceToHost = 0, cudaMemcpyHostToDevice = 1 };
#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2
#define cudaErrorInvalidDevicePointer 17
#define cudaHostRegisterPortable 1
inline const char* cudaGetErrorString(cudaError_t) { return "mock cudaSuccess"; }
inline const char* cudaGetErrorName(cudaError_t) { return "cudaSuccess"; }
inline int& tms_shim_host_register_calls() { static int c = 0; return c; }
inline int& tms_shim_host_unregister_calls() { static int c = 0; return c; }
inline cudaError_t cudaHostRegister(void*, size_t, unsigned int) { ++tms_shim_host_register_calls(); return cudaSuccess; }
inline cudaError_t cudaHostUnregister(void*) { ++tms_shim_host_unregister_calls(); return cudaSuccess; }
inline int& tms_shim_last_memcpy_device() { static thread_local int d = -1; return d; }
inline int& tms_shim_last_sync_device() { static thread_local int d = -1; return d; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    tms_shim_last_memcpy_device() = tms_shim_current_device();
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
inline cudaError_t cudaMallocHost(void**, size_t) { return cudaErrorMemoryAllocation; }
inline cudaError_t cudaFreeHost(void* ptr) { std::free(ptr); return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* dev) { *dev = tms_shim_current_device(); return cudaSuccess; }
inline cudaError_t cudaSetDevice(int dev) {
    tms_shim_current_device() = dev;
    tms_shim_current_ctx() = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(0x100 + dev + 1));
    return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() { tms_shim_last_sync_device() = tms_shim_current_device(); return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(size_t* free_b, size_t* total_b) { *free_b = 1ull << 30; *total_b = 2ull << 30; return cudaSuccess; }
inline cudaError_t cudaDeviceGetPCIBusId(char*, int, int) { return cudaErrorInvalidDevicePointer; }
''', encoding="utf-8")


def test_disk_backend_open_close_fork_and_core_lifecycle(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shim_dir = tmp_path / "cuda_shim"
    shim_dir.mkdir()
    _write_cuda_shims(shim_dir)

    source = tmp_path / "backend_core.cpp"
    source.write_text(r'''
#include "csrc/disk_backend.cpp"
#include "csrc/api_forwarder.h"

namespace APIForwarder {
cudaError_t call_real_cuda_malloc(void** ptr, size_t) { *ptr = nullptr; return cudaSuccess; }
cudaError_t call_real_cuda_free(void*) { return cudaSuccess; }
}

#include "csrc/core.cpp"
#include <cassert>
#include <cstdarg>
#include <dirent.h>
#include <sys/wait.h>
#include <memory>

static int g_expected_numa_node = 0;
static bool g_fail_next_probe_close = false;

static int pick_numa_node() {
    DIR* dir = opendir("/sys/devices/system/node");
    if (!dir) return -1;
    int best = -1;
    while (struct dirent* ent = readdir(dir)) {
        if (std::strncmp(ent->d_name, "node", 4) != 0 || ent->d_name[4] < '0' || ent->d_name[4] > '9') continue;
        int v = std::atoi(ent->d_name + 4);
        std::string full = std::string("/sys/devices/system/node/") + ent->d_name;
        struct stat st{};
        if (stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && (best < 0 || v < best)) best = v;
    }
    closedir(dir);
    return best;
}

extern "C" int __real_close(int fd);
extern "C" int __wrap_close(int fd) {
    if (g_fail_next_probe_close) {
        int fl = fcntl(fd, F_GETFL);
        if (fl >= 0 && (fl & O_DIRECT)) {
            g_fail_next_probe_close = false;
            __real_close(fd);
            errno = EIO;
            return -1;
        }
    }
    return __real_close(fd);
}

extern "C" long __real_syscall(long, ...);
extern "C" long __wrap_syscall(long number, ...) {
    va_list ap;
    va_start(ap, number);
    long rc = 0;
    if (number == SYS_mbind) {
        rc = 0;
    } else if (number == SYS_move_pages) {
        (void)va_arg(ap, int);
        unsigned long count = va_arg(ap, unsigned long);
        (void)va_arg(ap, void**);
        (void)va_arg(ap, const void*);
        int* status = va_arg(ap, int*);
        for (unsigned long i = 0; i < count; ++i) status[i] = g_expected_numa_node;
        rc = 0;
    } else if (number == SYS_statx) {
        int dirfd = va_arg(ap, int);
        const char* path = va_arg(ap, const char*);
        int flags = va_arg(ap, int);
        unsigned int mask = va_arg(ap, unsigned int);
        void* buf = va_arg(ap, void*);
        rc = __real_syscall(number, dirfd, path, flags, mask, buf);
    } else {
        errno = ENOSYS;
        rc = -1;
    }
    va_end(ap);
    return rc;
}

static int count_open_slot_fds(const std::string& proc_dir) {
    int count = 0;
    DIR* dir = opendir("/proc/self/fd");
    assert(dir != nullptr);
    while (struct dirent* ent = readdir(dir)) {
        if (ent->d_name[0] < '0' || ent->d_name[0] > '9') continue;
        char link_buf[4224];
        std::string p = std::string("/proc/self/fd/") + ent->d_name;
        ssize_t len = readlink(p.c_str(), link_buf, sizeof(link_buf));
        if (len > 0 && std::string(link_buf, static_cast<size_t>(len)).rfind(proc_dir, 0) == 0) {
            ++count;
        }
    }
    closedir(dir);
    return count;
}

int main(int, char** argv) {
    int node = pick_numa_node();
    if (node < 0) return 77;
    g_expected_numa_node = node;
    const size_t chunk_bytes = std::max<size_t>(65536, static_cast<size_t>(sysconf(_SC_PAGESIZE)));

    // Roll back cleanly on configure() close error and reject tmpfs before creating subdirs.
    {
        DiskBackend rb("", chunk_bytes);
        g_fail_next_probe_close = true;
        bool threw = false;
        try { rb.configure(argv[1], node); } catch (const std::runtime_error&) { threw = true; }
        assert(threw);
        assert(tms_shim_host_register_calls() == 1 && tms_shim_host_unregister_calls() == 1);
        assert(rb.info().find("\"direct\":false") != std::string::npos);
        assert(std::filesystem::is_empty(argv[1]));

        struct statfs shm_fs{};
        if (access("/dev/shm", W_OK | X_OK) == 0 && statfs("/dev/shm", &shm_fs) == 0 && shm_fs.f_type == 0x01021994) {
            std::string shm_sub = std::string("/dev/shm/tms_tmpfs_") + std::to_string(getpid());
            bool tmpfs_threw = false;
            try { rb.configure(shm_sub, node); } catch (const std::runtime_error&) { tmpfs_threw = true; }
            assert(tmpfs_threw && !std::filesystem::exists(shm_sub));
        }
    }

    // Named open-transfer-close zero-fd lifecycle + fork() child isolation + ~DiskBackend cleanup.
    std::string proc_dir_copy;
    {
        auto backend = std::make_unique<DiskBackend>("", chunk_bytes);
        backend->configure(argv[1], node);
        proc_dir_copy = backend->proc_dir_;
        assert(std::filesystem::exists(proc_dir_copy));
        assert(count_open_slot_fds(proc_dir_copy) == 0);

        const size_t size = chunk_bytes + 123;
        std::vector<uint8_t> buf(size);
        for (size_t i = 0; i < size; ++i) buf[i] = static_cast<uint8_t>((i * 31 + 7) & 0xff);

        DiskBackupSlot slot;
        backend->offload(buf.data(), buf.size(), slot);
        assert(slot.populated && std::filesystem::exists(slot.path));
        assert(count_open_slot_fds(proc_dir_copy) == 0);

        // Child process sees MADV_DONTFORK, zero open slot fds, and ~DiskBackend() does not touch parent dir/staging.
        pid_t child = fork();
        assert(child >= 0);
        if (child == 0) {
            assert(count_open_slot_fds(proc_dir_copy) == 0);
            int unreg_before = tms_shim_host_unregister_calls();
            backend.reset();
            assert(tms_shim_host_unregister_calls() == unreg_before);
            assert(std::filesystem::exists(slot.path));
            _exit(0);
        }
        int st = 0;
        assert(waitpid(child, &st, 0) == child && WIFEXITED(st) && WEXITSTATUS(st) == 0);

        std::fill(buf.begin(), buf.end(), 0);
        backend->onload(buf.data(), buf.size(), slot);
        assert(count_open_slot_fds(proc_dir_copy) == 0);
        for (size_t i = 0; i < size; ++i) assert(buf[i] == static_cast<uint8_t>((i * 31 + 7) & 0xff));

        std::string saved_path = slot.path;
        backend->release(slot);
        assert(!slot.populated && slot.path.empty() && !std::filesystem::exists(saved_path));
        backend.reset();
        assert(!std::filesystem::exists(proc_dir_copy));
    }

    // Core mutable resume() deletes disk slot immediately; write_once skips repeat writes;
    // ScopedCudaDevice restores caller CUcontext/device across cross-device pause/resume/free.
    {
        auto& tms = TorchMemorySaver::instance();
        tms.configure_direct_disk_backup(argv[2], node);
        CUcontext custom_ctx_dev0 = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(0x201));
        tms_shim_current_ctx() = custom_ctx_dev0;
        tms_shim_current_device() = 0;

        void* mut_ptr = nullptr;
        void* ro_ptr = nullptr;
        assert(tms.malloc(&mut_ptr, 1, 4096, "mut", false, CpuBackupKind::PINNED, true, false) == cudaSuccess);
        assert(tms.malloc(&ro_ptr, 1, 4096, "ro", false, CpuBackupKind::PINNED, true, true) == cudaSuccess);
        std::memset(mut_ptr, 0x3a, 4096);
        std::memset(ro_ptr, 0x5c, 4096);

        tms_shim_last_memcpy_device() = -1;
        assert(tms.pause("") == cudaSuccess);
        assert(tms_shim_last_memcpy_device() == 1);
        assert(tms_shim_current_device() == 0 && tms_shim_current_ctx() == custom_ctx_dev0);
        assert(tms.allocation_metadata_.at(mut_ptr).disk.populated);
        assert(tms.allocation_metadata_.at(ro_ptr).disk.populated);

        std::memset(mut_ptr, 0x11, 4096);
        std::memset(ro_ptr, 0x22, 4096);
        tms_shim_last_memcpy_device() = -1;
        assert(tms.resume("") == cudaSuccess);
        assert(tms_shim_last_memcpy_device() == 1);
        assert(tms_shim_current_device() == 0 && tms_shim_current_ctx() == custom_ctx_dev0);
        assert(static_cast<uint8_t*>(mut_ptr)[0] == 0x3a);
        assert(static_cast<uint8_t*>(ro_ptr)[0] == 0x5c);
        // Mutable slot is released on resume(); write_once slot stays populated until free().
        assert(!tms.allocation_metadata_.at(mut_ptr).disk.populated);
        assert(tms.allocation_metadata_.at(ro_ptr).disk.populated);

        std::memset(ro_ptr, 0x99, 4096);
        assert(tms.pause("ro") == cudaSuccess);
        assert(tms.get_disk_backup_info().find("\"readonly_skipped_pauses\":1") != std::string::npos);
        assert(tms.resume("ro") == cudaSuccess);
        assert(static_cast<uint8_t*>(ro_ptr)[0] == 0x5c);

        tms_shim_last_sync_device() = -1;
        assert(tms.free(mut_ptr) == cudaSuccess);
        assert(tms.free(ro_ptr) == cudaSuccess);
        assert(tms_shim_last_sync_device() == 1);
        assert(tms_shim_current_device() == 0 && tms_shim_current_ctx() == custom_ctx_dev0);
        assert(tms.get_disk_backup_info().find("\"active_slots\":0") != std::string::npos);
        tms_shim_current_ctx() = nullptr;
    }
    return 0;
}
''', encoding="utf-8")
    binary = tmp_path / "backend_core"
    _run_cmd(
        ["g++", "-std=c++17", "-fno-access-control", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
         "-pthread", "-DUSE_CUDA", "-I", str(shim_dir), "-I", str(root), str(source),
         "-Wl,--wrap=close", "-Wl,--wrap=syscall", "-o", str(binary)]
    )
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMS_TEST_DISK_DIR")) as disk_dir:
        d1 = Path(disk_dir) / "b1"
        d2 = Path(disk_dir) / "b2"
        d1.mkdir()
        d2.mkdir()
        env = {k: v for k, v in os.environ.items() if k != "TMS_DISK_BACKUP_DIR"}
        proc = _run_cmd([str(binary), str(d1), str(d2)], env=env, check=False)
        if proc.returncode == 77:
            pytest.skip("no sysfs NUMA node directory")
        assert proc.returncode == 0, proc.stdout + proc.stderr


def test_fail_fast_cdll_signatures_and_pep440_gate(monkeypatch):
    import torch
    import torch_memory_saver.entrypoint as entrypoint_mod
    from torch_memory_saver.binary_wrapper import BinaryWrapper, _setup_function_signatures
    from torch_memory_saver.entrypoint import (
        _BackupTarget,
        _TorchMemorySaverImpl,
        _torch_supports_detached_mempool_free,
    )

    monkeypatch.setattr(entrypoint_mod.sys, "platform", "linux")
    monkeypatch.setattr(entrypoint_mod, "is_mlu", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None)

    for missing in ("tms_set_write_once", "tms_configure_direct_disk_backup", "tms_set_disk_backup_dir_checked"):
        all_names = {
            name: SimpleNamespace()
            for name in (
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
            if name != missing
        }
        with pytest.raises(AttributeError):
            _setup_function_signatures(SimpleNamespace(**all_names))

    blocked_vers = (
        "2.6.0+cu124", "2.7.1", "2.8.0a0+5228986c39", "2.8.0b1",
        "2.8.0rc1", "2.8.0-rc1", "2.8.0preview1", "2.8.0.dev20250201", "invalid-ver",
    )
    allowed_vers = ("2.8", "2.8.0", "2.8.0+cu128", "2.8.0.post1", "2.8.1a0+git1234", "1!2.0.0")
    with monkeypatch.context() as mp:
        for v in blocked_vers:
            mp.setattr(torch, "__version__", v)
            assert not _torch_supports_detached_mempool_free(), v
        for v in allowed_vers:
            mp.setattr(torch, "__version__", v)
            assert _torch_supports_detached_mempool_free(), v

    impl = _TorchMemorySaverImpl.__new__(_TorchMemorySaverImpl)
    impl._is_xpu = False
    impl._is_musa = False
    impl._hook_mode = "preload"
    impl._open_write_once_counts = {}
    impl._active_mem_pools = set()
    impl._binary_wrapper = SimpleNamespace(cdll=SimpleNamespace(tms_supports_write_once=lambda: False))
    with pytest.raises(NotImplementedError, match="write_once is not supported"):
        with impl.region(tag="t", backup=_BackupTarget.CPU, cpu_backup_backend=None, write_once=True):
            pass

    wrapper = BinaryWrapper.__new__(BinaryWrapper)
    wrapper.cdll = impl._binary_wrapper.cdll
    with pytest.raises(NotImplementedError, match="write_once is not supported"):
        wrapper.set_config(tag="t", interesting_region=True, enable_cpu_backup=True, write_once=True)

    with monkeypatch.context() as mp:
        mp.setattr(torch.version, "hip", "6.2.41133")
        with pytest.raises(NotImplementedError, match="write_once is not supported"):
            wrapper.set_config(tag="t", interesting_region=True, enable_cpu_backup=True, write_once=True)

    impl._is_musa = True
    with pytest.raises(NotImplementedError, match="direct disk backup"):
        with impl.region(tag="t", backup=_BackupTarget.DISK, cpu_backup_backend=None):
            pass


def _make_stateful_cdll():
    class _StatefulCdll:
        def __init__(self):
            self._tls = threading.local()
            self.direct = False
            self.dir = ""
            self.pause_calls = []

        def _s(self):
            d = getattr(self._tls, "d", None)
            if d is None:
                d = {
                    "tag": b"default", "interesting": False, "cpu": False,
                    "backend": b"pinned", "disk": False, "write_once": False,
                }
                self._tls.d = d
            return d

        def tms_set_current_tag(self, v): self._s()["tag"] = v
        def tms_get_current_tag(self): return self._s()["tag"]
        def tms_set_interesting_region(self, v): self._s()["interesting"] = v
        def tms_get_interesting_region(self): return self._s()["interesting"]
        def tms_set_enable_cpu_backup(self, v): self._s()["cpu"] = v
        def tms_get_enable_cpu_backup(self): return self._s()["cpu"]
        def tms_set_cpu_backup_backend(self, v): self._s()["backend"] = v
        def tms_get_cpu_backup_backend(self): return self._s()["backend"]
        def tms_set_enable_disk_backup(self, v): self._s()["disk"] = v
        def tms_get_enable_disk_backup(self): return self._s()["disk"]
        def tms_set_write_once(self, v): self._s()["write_once"] = v
        def tms_get_write_once(self): return self._s()["write_once"]
        def tms_supports_write_once(self): return True
        def tms_configure_direct_disk_backup(self, p, _n):
            self.direct = True
            self.dir = os.fsdecode(os.path.realpath(p))
            return None
        def tms_set_disk_backup_dir_checked(self, p):
            self.dir = os.fsdecode(p)
            return None
        def tms_get_disk_backup_info(self):
            return json.dumps({"direct": self.direct, "directory": self.dir}).encode()
        def tms_pause(self, tag=None):
            self.pause_calls.append(tag)
            return 0
        def tms_affected_devices(self, _tag=None, _ids=None, _cap=0):
            return 0

    return _StatefulCdll()


def test_region_validation_pool_separation_and_lifecycle(tmp_path, monkeypatch):
    import torch
    import torch_memory_saver.entrypoint as entrypoint_mod
    from torch_memory_saver.binary_wrapper import BinaryWrapper
    from torch_memory_saver.entrypoint import (
        TorchMemorySaver,
        _BackupTarget,
        _PoolKey,
        _StorageMode,
        _TorchMemorySaverImpl,
        _resolve_region_backup,
    )

    monkeypatch.setattr(entrypoint_mod.sys, "platform", "linux")
    monkeypatch.setattr(entrypoint_mod, "is_mlu", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch, "__version__", "2.8.0")

    assert _resolve_region_backup(
        backup="cpu_mmap", write_once=True,
        enable_cpu_backup=None, enable_disk_backup=None, readonly_disk_backup=None,
    ) == (_BackupTarget.CPU, "mmap", True)
    assert _resolve_region_backup(
        backup=None, write_once=None,
        enable_cpu_backup=None, enable_disk_backup=True, readonly_disk_backup=True,
    ) == (_BackupTarget.DISK, None, True)

    tms = TorchMemorySaver()
    for bad_kwargs in (
        dict(backup="invalid"),
        dict(backup="cpu_mmap", cpu_backup_backend="pinned"),
        dict(backup="cpu", enable_cpu_backup=True),
        dict(backup="disk", write_once=True, readonly_disk_backup=True),
        dict(backup="cpu", readonly_disk_backup=True),
        dict(write_once=True),
        dict(backup="none", write_once=True),
        dict(backup="cpu", write_once="false"),
        dict(tag="bad\x00tag"),
    ):
        with pytest.raises(ValueError):
            with tms.region(**bad_kwargs):
                pass

    stale_root = tmp_path / "stale_spill"
    stale_sub = stale_root / "pid_999999"
    stale_sub.mkdir(parents=True)
    (stale_sub / "slot_1.bin").write_bytes(b"stale")
    tms.clear_disk_backup_dir(str(stale_root))
    assert list(stale_root.iterdir()) == []

    trace_lock = threading.Lock()
    live_active_pools = []
    destroyed_snapshots = []
    destructor_violations = []
    pool_seq = [0]

    class _TrackedPool:
        def __init__(self, name): self.name = name
        def __del__(self):
            with trace_lock:
                snap = list(live_active_pools)
                destroyed_snapshots.append((self.name, snap))
                if any(d == 0 for _, d in snap):
                    destructor_violations.append((self.name, snap))

    def _new_pool():
        with trace_lock:
            pool_seq[0] += 1
            return _TrackedPool(f"pool_{pool_seq[0]}")

    class _ExclusiveCtx:
        def __init__(self, pool, dev): self.pool, self.dev = pool, dev
        def __enter__(self):
            with trace_lock:
                assert not any(d == self.dev for _, d in live_active_pools)
                live_active_pools.append((self.pool, self.dev))
            return self
        def __exit__(self, *_a):
            with trace_lock:
                live_active_pools.remove((self.pool, self.dev))
            return False

    cdll = _make_stateful_cdll()
    wrapper = BinaryWrapper.__new__(BinaryWrapper)
    wrapper.cdll = cdll
    wrapper._resolved_binary_path = "/fake/hook_torch.so"

    impl = _TorchMemorySaverImpl.__new__(_TorchMemorySaverImpl)
    impl._is_xpu = False
    impl._is_musa = False
    impl._hook_mode = "torch"
    impl._open_write_once_counts = {}
    impl._active_mem_pools = set()
    impl._mem_pools = defaultdict(_new_pool)
    impl._write_once_pools = defaultdict(_new_pool)
    impl._device_module = SimpleNamespace(
        current_device=lambda: 0,
        MemPool=_new_pool,
        use_mem_pool=lambda pool: _ExclusiveCtx(pool, 0),
    )
    impl._binary_wrapper = wrapper
    tms._impl = impl
    _TorchMemorySaverImpl._instances.add(impl)

    try:
        # Reject clearing the active configured disk directory in-process.
        tms.configure_direct_disk_backup(str(stale_root))
        with pytest.raises(RuntimeError, match="active configured disk backup directory"):
            tms.clear_disk_backup_dir(str(stale_root))

        # Mutable and write_once pools live in separate maps; pause(tag) retires only _write_once_pools.
        with tms.region(tag="shared", backup="cpu_mmap"):
            mut_pool = impl._mem_pools[_PoolKey("shared", 0, _StorageMode.CPU_MMAP)]
        with tms.region(tag="shared", backup="cpu", write_once=True):
            wo_pool = impl._write_once_pools[_PoolKey("shared", 0, _StorageMode.CPU_PINNED)]
        assert mut_pool is not wo_pool
        tms.pause("shared")
        assert _PoolKey("shared", 0, _StorageMode.CPU_MMAP) in impl._mem_pools
        assert _PoolKey("shared", 0, _StorageMode.CPU_PINNED) not in impl._write_once_pools

        # Nested region()/disable() suspends outer MemPool cleanly; write_once cannot nest.
        with tms.region(tag="outer", backup="none"):
            outer_p = impl._mem_pools[_PoolKey("outer", 0, _StorageMode.NONE)]
            with tms.region(tag="inner", backup="cpu_mmap"):
                inner_p = impl._mem_pools[_PoolKey("inner", 0, _StorageMode.CPU_MMAP)]
                assert live_active_pools == [(inner_p, 0)]
                with tms.disable():
                    assert len(live_active_pools) == 1 and live_active_pools[0][0] not in (outer_p, inner_p)
                assert live_active_pools == [(inner_p, 0)]
            assert live_active_pools == [(outer_p, 0)]
        assert live_active_pools == []

        with tms.region(tag="frozen", backup="cpu", write_once=True):
            with pytest.raises(RuntimeError, match="open write_once region"):
                tms.pause("frozen")
            with pytest.raises(RuntimeError, match="inside an active write_once=True region"):
                with tms.region(tag="sub", backup="cpu"):
                    pass
            with pytest.raises(RuntimeError, match="inside an active write_once=True region"):
                with tms.disable():
                    pass

        # Real lock contention between pause() sync and concurrent region() entry + deferred retirement.
        sync_entered = threading.Event()
        allow_sync = threading.Event()
        enter_contended = threading.Event()
        real_lock = _TorchMemorySaverImpl._region_pause_lock
        pause_tid = [None]
        errors = []

        class _ContendedLock:
            def __enter__(self):
                if threading.get_ident() != pause_tid[0]:
                    if not real_lock.acquire(blocking=False):
                        enter_contended.set()
                        real_lock.acquire()
                else:
                    real_lock.acquire()
                return self
            def __exit__(self, *_a):
                real_lock.release()
                return False

        orig_sync = impl._sync_affected_devices
        impl._sync_affected_devices = lambda _t: (sync_entered.set(), allow_sync.wait(timeout=5))
        _TorchMemorySaverImpl._region_pause_lock = _ContendedLock()
        try:
            with tms.region(tag="race", backup="cpu", write_once=True):
                old_wo = impl._write_once_pools[_PoolKey("race", 0, _StorageMode.CPU_PINNED)]
            t_pause = threading.Thread(target=lambda: (pause_tid.__setitem__(0, threading.get_ident()), tms.pause("race")))
            t_pause.start()
            assert sync_entered.wait(timeout=5)
            entered_flag = threading.Event()
            def _race_enter():
                try:
                    with tms.region(tag="race", backup="cpu", write_once=True):
                        entered_flag.set()
                except BaseException as exc:
                    errors.append(exc)
            th = threading.Thread(target=_race_enter)
            th.start()
            assert enter_contended.wait(timeout=5)
            assert not entered_flag.is_set()
            allow_sync.set()
            t_pause.join(timeout=5)
            th.join(timeout=5)
            assert errors == [] and entered_flag.is_set()
            assert impl._write_once_pools[_PoolKey("race", 0, _StorageMode.CPU_PINNED)] is not old_wo
        finally:
            impl._sync_affected_devices = orig_sync
            _TorchMemorySaverImpl._region_pause_lock = real_lock

        # pause() while unrelated region is active defers retired write_once MemPool destruction until exit.
        with tms.region(tag="defer_wo", backup="cpu", write_once=True):
            pass
        defer_name = impl._write_once_pools[_PoolKey("defer_wo", 0, _StorageMode.CPU_PINNED)].name
        with tms.region(tag="holder", backup="cpu"):
            before_n = len(destroyed_snapshots)
            tms.pause("defer_wo")
            assert len(destroyed_snapshots) == before_n
        assert destroyed_snapshots[-1] == (defer_name, [])
        assert destructor_violations == []
    finally:
        _TorchMemorySaverImpl._instances.discard(impl)
        impl._clear_all_pools()


def test_built_extensions_manylinux2014_and_symbol_visibility(tmp_path, monkeypatch):
    import torch_memory_saver

    root = Path(__file__).resolve().parents[1]
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "click", None)
        mp.setitem(sys.modules, "typer", None)
        spec = importlib.util.spec_from_file_location(
            "release_checks_no_cli",
            root / ".claude/skills/tms-publish-release/scripts/release_checks.py",
        )
        release_checks = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, release_checks)
        spec.loader.exec_module(release_checks)

    pkg_dir = Path(torch_memory_saver.__file__).resolve().parent
    binaries = sorted(
        {p.resolve() for search in (root, pkg_dir, pkg_dir.parent) if search.is_dir()
         for p in search.glob("torch_memory_saver_hook_mode_*.so") if p.is_file()}
    )
    assert binaries, "expected built torch_memory_saver_hook_mode_*.so binaries"

    nm_bin = shutil.which("nm")
    for binary in binaries:
        data = binary.read_bytes()
        undef = release_checks._read_elf_undefined_symbols(binary=data, source=binary.name)
        defined = release_checks._read_elf_defined_symbols(binary=data, source=binary.name)
        assert not release_checks._find_forbidden_glibc_symbols(undef), binary.name
        assert not release_checks._find_leaked_internal_symbols(defined | undef), binary.name
        if nm_bin:
            nm_u = {ln.split()[-1] for ln in _run_cmd([nm_bin, "-D", "-u", str(binary)]).stdout.splitlines() if ln.split()}
            nm_d = {ln.split()[-1] for ln in _run_cmd([nm_bin, "-D", "--defined-only", str(binary)]).stdout.splitlines() if ln.split()}
            assert not release_checks._find_forbidden_glibc_symbols(nm_u), binary.name
            assert not release_checks._find_leaked_internal_symbols(nm_u | nm_d), binary.name

    # Built dist/*.whl (if present) must also pass manylinux2014 ABI and symbol checks.
    for wheel_path in sorted((root / "dist").glob("*.whl")):
        with zipfile.ZipFile(wheel_path) as whl:
            for member in (n for n in whl.namelist() if n.endswith(".so")):
                data = whl.read(member)
                undef = release_checks._read_elf_undefined_symbols(binary=data, source=member)
                defined = release_checks._read_elf_defined_symbols(binary=data, source=member)
                assert not release_checks._find_forbidden_glibc_symbols(undef), member
                if "manylinux2014" in wheel_path.name:
                    assert not release_checks._find_post_manylinux2014_symbol_versions(
                        binary=data, source=member, undefined_symbols=undef
                    ), member
                assert not release_checks._find_leaked_internal_symbols(defined | undef), member

    # LD_PRELOAD of legacy internal C++ symbols must not interpose on hook_mode='torch'.
    torch_sos = [p for p in binaries if "hook_mode_torch" in p.name]
    assert torch_sos, "expected built torch_memory_saver_hook_mode_torch*.so"

    stub_dir = tmp_path / "stub_cuda_rt"
    stub_dir.mkdir()
    stub_src = stub_dir / "stub_cuda.c"
    stub_src.write_text(r'''
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
int cuDeviceGet(int* d, int o) { if (d) *d = o; return 0; }
int cuDeviceGetAttribute(int* p, int a, int d) { (void)a; (void)d; if (p) *p = 1; return 0; }
int cuGetErrorString(int e, const char** s) { (void)e; if (s) *s = "ok"; return 0; }
int cuMemGetAllocationGranularity(size_t* g, const void* p, int t) { (void)p; (void)t; if (g) *g = 4096; return 0; }
int cuMemCreate(unsigned long long* h, size_t s, const void* p, unsigned long long f) { (void)s; (void)p; (void)f; if (h) *h = 1; return 0; }
int cuMemAddressReserve(unsigned long long* ptr, size_t sz, size_t a, unsigned long long addr, unsigned long long f) {
    (void)a; (void)addr; (void)f; void* m = calloc(1, sz ? sz : 1); if (!m) return 2; *ptr = (unsigned long long)m; return 0;
}
int cuMemMap(unsigned long long p, size_t s, size_t o, unsigned long long h, unsigned long long f) { (void)p; (void)s; (void)o; (void)h; (void)f; return 0; }
int cuMemSetAccess(unsigned long long p, size_t s, const void* d, size_t c) { (void)p; (void)s; (void)d; (void)c; return 0; }
int cuMemUnmap(unsigned long long p, size_t s) { (void)p; (void)s; return 0; }
int cuMemRelease(unsigned long long h) { (void)h; return 0; }
int cuMemAddressFree(unsigned long long p, size_t s) { (void)s; free((void*)p); return 0; }
static _Thread_local void* g_ctx = NULL;
int cuCtxGetCurrent(void** c) { if (c) *c = g_ctx; return 0; }
int cuCtxSetCurrent(void* c) { g_ctx = c; return 0; }
int cuCtxGetDevice(int* d) { if (d) *d = 0; return 0; }
const char* cudaGetErrorString(int e) { (void)e; return "ok"; }
const char* cudaGetErrorName(int e) { (void)e; return "cudaSuccess"; }
int cudaGetDevice(int* d) { if (d) *d = 0; return 0; }
int cudaSetDevice(int d) { (void)d; return 0; }
int cudaDeviceSynchronize(void) { return 0; }
int cudaDeviceGetPCIBusId(char* b, int l, int d) { (void)d; if (b && l > 0) b[0] = '\0'; return 1; }
int cudaHostRegister(void* p, size_t s, unsigned int f) { (void)p; (void)s; (void)f; return 0; }
int cudaHostUnregister(void* p) { (void)p; return 0; }
int cudaMallocHost(void** p, size_t s) { *p = calloc(1, s ? s : 1); return *p ? 0 : 2; }
int cudaFreeHost(void* p) { free(p); return 0; }
int cudaMemcpy(void* dst, const void* src, size_t n, int k) { (void)k; memcpy(dst, src, n); return 0; }
int cudaMemGetInfo(size_t* f, size_t* t) { if (f) *f = 1ULL << 30; if (t) *t = 2ULL << 30; return 0; }
int cudaMalloc(void** p, size_t s) { *p = calloc(1, s ? s : 1); return *p ? 0 : 2; }
int cudaFree(void* p) { free(p); return 0; }
''', encoding="utf-8")
    _run_cmd(["gcc", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
              "-Wl,-soname,libcuda.so.1", str(stub_src), "-o", str(stub_dir / "libcuda.so.1")])
    for ver in ("12", "13"):
        vs = stub_dir / f"cudart{ver}.ver"
        vs.write_text(f"libcudart.so.{ver} {{ global: *; }};\n", encoding="utf-8")
        _run_cmd(["gcc", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
                  f"-Wl,-soname,libcudart.so.{ver}", f"-Wl,--version-script={vs}",
                  str(stub_src), "-o", str(stub_dir / f"libcudart.so.{ver}")])

    fake_src = tmp_path / "fake_legacy_preload.cpp"
    fake_src.write_text(r'''
#include <cstdint>
static int g_calls = 0;
static uint64_t g_fake = 0;
class TorchMemorySaver { public: __attribute__((visibility("default"))) static TorchMemorySaver& instance(); };
TorchMemorySaver& TorchMemorySaver::instance() { ++g_calls; return *reinterpret_cast<TorchMemorySaver*>(&g_fake); }
extern "C" __attribute__((visibility("default"))) int tms_fake_legacy_called() { return g_calls; }
''', encoding="utf-8")
    fake_so = tmp_path / "fake_legacy_preload.so"
    _run_cmd(["g++", "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", str(fake_src), "-o", str(fake_so)])

    probe = (
        "import ctypes, json, sys\n"
        "fake = ctypes.CDLL(None); fake.tms_fake_legacy_called.restype = ctypes.c_int\n"
        "for so_path in sys.argv[1:]:\n"
        "    cdll = ctypes.CDLL(so_path)\n"
        "    cdll.tms_get_disk_backup_info.restype = ctypes.c_char_p\n"
        "    cdll.tms_set_interesting_region.argtypes = [ctypes.c_bool]\n"
        "    cdll.tms_torch_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]\n"
        "    cdll.tms_torch_malloc.restype = ctypes.c_void_p\n"
        "    cdll.tms_torch_free.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]\n"
        "    assert json.loads(cdll.tms_get_disk_backup_info().decode())['direct'] is False\n"
        "    cdll.tms_set_interesting_region(True)\n"
        "    ptr = cdll.tms_torch_malloc(4096, 0, None); assert ptr\n"
        "    cdll.tms_torch_free(ptr, 4096, 0, None)\n"
        "    cdll.tms_set_interesting_region(False)\n"
        "assert fake.tms_fake_legacy_called() == 0\n"
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["LD_LIBRARY_PATH"] = f"{stub_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    env["LD_PRELOAD"] = f"{fake_so}:{env.get('LD_PRELOAD', '')}".rstrip(":")
    env.pop("TMS_DISK_BACKUP_DIR", None)
    _run_cmd([sys.executable, "-c", probe, *map(str, torch_sos)], env=env)
