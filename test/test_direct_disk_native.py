"""CPU-only tests of the real direct-I/O loop (no GPU, registration or install)."""
import os
from pathlib import Path
import subprocess
import tempfile



def test_native_direct_io(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "direct.cpp"
    source.write_text(r'''
#include "csrc/disk_backend.cpp"
#include <cassert>
static int write_fault = 0, read_fault = 0;
static size_t short_alignment = 0;
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
    char name[4096];
    snprintf(name, sizeof(name), "%s/native_XXXXXX", argv[1]);
    File file(mkostemp(name, O_DIRECT | O_CLOEXEC));
    system_check(file.fd >= 0, "create direct fixture");
    struct statx sx = {};
    system_check(statx(file.fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0, "statx");
    if (!(sx.stx_mask & STATX_DIOALIGN)) {
        // Test fixture only: exercise the I/O loop on older local filesystems.
        // Production configuration still rejects a missing alignment report.
        sx.stx_dio_mem_align = sx.stx_dio_offset_align = 4096;
    }
    assert(fcntl(file.fd, F_GETFL) & O_DIRECT);
    assert(fcntl(file.fd, F_GETFD) & FD_CLOEXEC);
    size_t align = std::lcm(sx.stx_dio_mem_align, sx.stx_dio_offset_align);
    void* buffer = nullptr;
    assert(posix_memalign(&buffer, std::max(align, sizeof(void*)), 4 * align) == 0);
    for (size_t logical : {size_t(1), align - 1, align, align + 1, 3 * align - 1}) {
        size_t padded = round_up(logical, sx.stx_dio_offset_align);
        for (int cycle = 0; cycle < 3; ++cycle) {
            memset(buffer, 0, padded);
            memset(buffer, cycle + 51, logical);
            assert(ftruncate(file.fd, padded) == 0);
            direct_io(file.fd, buffer, padded, 0, sx.stx_dio_offset_align, sx.stx_dio_mem_align, true);
            memset(buffer, 0xff, padded);
            direct_io(file.fd, buffer, padded, 0, sx.stx_dio_offset_align, sx.stx_dio_mem_align, false);
            for (size_t i = 0; i < padded; ++i)
                assert(static_cast<unsigned char*>(buffer)[i] == (i < logical ? cycle + 51 : 0));
        }
    }
    short_alignment = align;
    memset(buffer, 0x73, 4 * align);
    write_fault = 1;
    direct_io(file.fd, buffer, 4 * align, 0, sx.stx_dio_offset_align, sx.stx_dio_mem_align, true);
    memset(buffer, 0, 4 * align);
    read_fault = 1;
    direct_io(file.fd, buffer, 4 * align, 0, sx.stx_dio_offset_align, sx.stx_dio_mem_align, false);
    assert(write_fault == 0 && read_fault == 0);
    for (size_t i = 0; i < 4 * align; ++i) assert(static_cast<unsigned char*>(buffer)[i] == 0x73);
    bool failed = false;
    assert(ftruncate(file.fd, 0) == 0);
    try { direct_io(file.fd, buffer, align, 0, sx.stx_dio_offset_align, sx.stx_dio_mem_align, false); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    failed = false;
    try { direct_io(file.fd, static_cast<char*>(buffer) + 1, align, 0,
                    sx.stx_dio_offset_align, sx.stx_dio_mem_align, true); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    DiskBackend backend(argv[1], align);
    failed = false;
    try { backend.configure(argv[1], -1); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    if (access("/dev/shm", F_OK) == 0) {
        failed = false;
        try { backend.configure("/dev/shm", 0); }
        catch (const std::runtime_error&) { failed = true; }
        assert(failed);
    }
    free(buffer);
    file.finish();
    assert(unlink(name) == 0);
}
''')
    binary = tmp_path / "direct"
    subprocess.run(["g++", "-std=c++17", "-DUSE_XPU", "-I", str(root),
                    str(source), "-Wl,--wrap=pwrite", "-Wl,--wrap=pread", "-o", str(binary)], check=True, capture_output=True, text=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMS_TEST_DISK_DIR")) as disk_dir:
        result = subprocess.run([str(binary), disk_dir], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


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
#ifdef MADV_NOHUGEPAGE
    if (madvise(buf, chunk_bytes, MADV_NOHUGEPAGE) != 0) {}
#endif
    memset(buf, 0x3a, chunk_bytes);
    if (mlock(buf, chunk_bytes) != 0) {}

    void* pages[4] = {
        buf,
        static_cast<uint8_t*>(buf) + page_bytes,
        static_cast<uint8_t*>(buf) + 2 * page_bytes,
        static_cast<uint8_t*>(buf) + 3 * page_bytes,
    };
    int real_status[4] = {-1, -1, -1, -1};
    assert(syscall(SYS_move_pages, 0, 4, pages, nullptr, real_status, 0) == 0);
    for (int i = 1; i < 4; ++i) assert(real_status[i] == real_status[0]);
    int real_node = real_status[0];

    verify_staging_pages_numa(buf, chunk_bytes, page_bytes, real_node, [](size_t count, void** p, int* st) {
        return syscall(SYS_move_pages, 0, count, p, nullptr, st, 0);
    });
    for (size_t i = 0; i < chunk_bytes; ++i) assert(static_cast<uint8_t*>(buf)[i] == 0x3a);

    int calls = 0;
    verify_staging_pages_numa(buf, chunk_bytes, page_bytes, 0, [&](size_t count, void**, int* st) {
        ++calls;
        for (size_t i = 0; i < count; ++i) st[i] = 0;
        if (calls <= 2) st[2] = -ENOENT;
        return 0;
    });
    assert(calls == 3);

    int persistent_calls = 0;
    bool caught_enoent = false;
    try {
        verify_staging_pages_numa(buf, chunk_bytes, page_bytes, 0, [&](size_t count, void**, int* st) {
            ++persistent_calls;
            for (size_t i = 0; i < count; ++i) st[i] = 0;
            st[1] = -ENOENT;
            return 0;
        });
    } catch (const std::runtime_error& e) {
        caught_enoent = std::string(e.what()).find("observed -2") != std::string::npos;
    }
    assert(caught_enoent && persistent_calls == 8);

    int wrong_node_calls = 0;
    bool caught_wrong_node = false;
    try {
        verify_staging_pages_numa(buf, chunk_bytes, page_bytes, 0, [&](size_t count, void**, int* st) {
            ++wrong_node_calls;
            for (size_t i = 0; i < count; ++i) st[i] = 0;
            st[0] = 1;
            return 0;
        });
    } catch (const std::runtime_error& e) {
        caught_wrong_node = std::string(e.what()).find("observed 1") != std::string::npos;
    }
    assert(caught_wrong_node && wrong_node_calls == 1);

    munmap(buf, chunk_bytes);
    return 0;
}
''')
    binary = tmp_path / "numa_retry"
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-DUSE_XPU", "-I", str(root),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_readonly_direct_disk_slot_reuse(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "readonly_slot.cpp"
    source.write_text(r'''
#include "csrc/disk_backend.cpp"
#include <cassert>
int main(int, char** argv) {
    DiskBackend backend(argv[1], 65536);
    DiskBackupSlot ro_slot, rw_slot;
    int ro_fd = backend.create_slot_file_(ro_slot, true);
    int rw_fd = backend.create_slot_file_(rw_slot, false);
    assert(ro_slot.path.find("/tms_ro_") != std::string::npos);
    assert(rw_slot.path.find("/tms_") != std::string::npos);
    assert(rw_slot.path.find("/tms_ro_") == std::string::npos);

    struct statx sx = {};
    system_check(statx(ro_fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0, "statx");
    if (!(sx.stx_mask & STATX_DIOALIGN)) {
        sx.stx_dio_mem_align = sx.stx_dio_offset_align = 4096;
    }
    size_t align = std::lcm(sx.stx_dio_mem_align, sx.stx_dio_offset_align);
    void* buf = nullptr;
    assert(posix_memalign(&buf, std::max(align, sizeof(void*)), align) == 0);
    memset(buf, 0x5c, align);
    assert(ftruncate(ro_fd, align) == 0);
    direct_io(ro_fd, buf, align, 0, sx.stx_dio_offset_align, sx.stx_dio_mem_align, true);
    assert(close(ro_fd) == 0);
    assert(close(rw_fd) == 0);

    backend.record_readonly_skip(align);
    std::string info = backend.info();
    assert(info.find("\"readonly_skipped_write_bytes\":" + std::to_string(align)) != std::string::npos);
    assert(info.find("\"readonly_skipped_pauses\":1") != std::string::npos);

    DiskBackupSlot reopen_ro{ro_slot.path, align, align};
    int fd2 = backend.open_slot_file_(reopen_ro);
    memset(buf, 0, align);
    direct_io(fd2, buf, align, 0, sx.stx_dio_offset_align, sx.stx_dio_mem_align, false);
    for (size_t i = 0; i < align; ++i) assert(static_cast<unsigned char*>(buf)[i] == 0x5c);
    assert(close(fd2) == 0);

    std::string ro_path = ro_slot.path;
    backend.release(ro_slot);
    backend.release(rw_slot);
    assert(access(ro_path.c_str(), F_OK) != 0);
    free(buf);
    return 0;
}
''')
    binary = tmp_path / "readonly_slot"
    subprocess.run(
        ["g++", "-std=c++17", "-fno-access-control", "-Wall", "-Wextra", "-Werror", "-DUSE_XPU",
         "-I", str(root), str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMS_TEST_DISK_DIR")) as disk_dir:
        result = subprocess.run([str(binary), disk_dir], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
