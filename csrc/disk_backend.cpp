#include "disk_backend.h"
#include "macro.h"
#include "utils.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>

namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("direct disk backup: " + message);
}
void system_check(bool ok, const char* operation) {
    if (!ok) throw std::runtime_error(std::string("direct disk backup: ") + operation + ": " + strerror(errno));
}
size_t round_up(size_t n, size_t a) {
    require(a && n <= std::numeric_limits<size_t>::max() - (a - 1), "size/alignment overflow");
    return (n + a - 1) / a * a;
}
struct File {
    int fd;
    explicit File(int value) : fd(value) {}
    ~File() { if (fd >= 0) close(fd); }
    void finish() { int value = fd; fd = -1; system_check(close(value) == 0, "close"); }
};
void direct_io(int fd, void* buf, size_t n, size_t offset, size_t alignment,
               size_t mem_alignment, bool write) {
    require(alignment && mem_alignment && n % alignment == 0 && offset % alignment == 0 &&
            reinterpret_cast<uintptr_t>(buf) % mem_alignment == 0,
            "unaligned direct I/O request; refusing buffered fallback");
    size_t done = 0;
    while (done < n) {
        ssize_t result = write
            ? pwrite(fd, static_cast<uint8_t*>(buf) + done, n - done, offset + done)
            : pread(fd, static_cast<uint8_t*>(buf) + done, n - done, offset + done);
        if (result < 0 && errno == EINTR) continue;
        system_check(result >= 0, write ? "O_DIRECT pwrite" : "O_DIRECT pread");
        require(result > 0, write ? "zero-length write" : "unexpected EOF");
        done += static_cast<size_t>(result);
        require(done == n || (done % alignment == 0 && done % mem_alignment == 0),
                "unaligned short direct I/O; refusing buffered fallback");
    }
}
std::string json_string(const std::string& value) {
    std::string out = "\"";
    const char* hex = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += c;
    }
    return out + '"';
}
#if defined(__linux__)
template <typename MovePagesFn>
void verify_staging_pages_numa(
    void* buf, size_t chunk_bytes, size_t page_bytes, int expected_node, MovePagesFn&& move_pages_fn
) {
    const size_t count = chunk_bytes / page_bytes;
    std::vector<void*> pages(count);
    std::vector<int> status(count, -1);
    for (size_t i = 0; i < count; ++i) pages[i] = static_cast<uint8_t*>(buf) + i * page_bytes;
    // SMMUv3/ATS cudaHostRegister leaves VmPin=0; reading a -ENOENT page waits on any
    // in-flight kernel compaction/THP migration PTE before re-checking NUMA residency.
    for (int attempt = 0; attempt < 8; ++attempt) {
        system_check(move_pages_fn(count, pages.data(), status.data()) == 0,
                     "move_pages query after cudaHostRegister");
        bool saw_enoent = false;
        for (size_t i = 0; i < count; ++i) {
            if (status[i] == -ENOENT && attempt + 1 < 8) {
                (void)*static_cast<const volatile uint8_t*>(pages[i]);
                saw_enoent = true;
            }
        }
        if (!saw_enoent) break;
        sched_yield();
    }
    for (int node : status) {
        require(node == expected_node,
                "staging page outside requested NUMA node (observed " + std::to_string(node) + ")");
    }
}
#endif
}  // namespace

std::string compute_disk_backup_dir_from_env() {
    const char* dir = std::getenv("TMS_DISK_BACKUP_DIR");
    return dir ? dir : "/tmp";
}
size_t compute_disk_chunk_bytes_from_env() {
    const char* value = std::getenv("TMS_DISK_BACKUP_CHUNK_MB");
    if (!value) return 256ull * 1024 * 1024;
    char* end = nullptr;
    errno = 0;
    unsigned long long mb = std::strtoull(value, &end, 10);
    require(errno == 0 && end != value && *end == '\0' && value[0] != '-' && mb > 0 &&
            mb <= std::numeric_limits<size_t>::max() / (1024 * 1024), "invalid TMS_DISK_BACKUP_CHUNK_MB");
    return mb * 1024 * 1024;
}
DiskBackend::DiskBackend(std::string dir, size_t chunk_bytes)
    : dir_(std::move(dir)), chunk_bytes_(chunk_bytes) {}
DiskBackend::~DiskBackend() {
#if defined(USE_CUDA) && defined(__linux__)
    if (staging_buf_) cudaHostUnregister(staging_buf_);
    if (mapping_base_) munmap(mapping_base_, mapping_bytes_);
#endif
}
void DiskBackend::set_dir(const std::string& dir) {
    require(!configured_ || dir == dir_, "placement is immutable after configuration");
    require(!allocation_started_, "directory must be set before disk allocation");
    dir_ = dir;
}
void DiskBackend::require_configured() {
    SIMPLE_CHECK(configured_, "call configure_direct_disk_backup(path, numa_node=...) before disk allocation");
    allocation_started_ = true;
}
void DiskBackend::check_fd_(int fd) {
    int flags = fcntl(fd, F_GETFL);
    int descriptor_flags = fcntl(fd, F_GETFD);
    system_check(flags >= 0 && descriptor_flags >= 0, "fcntl");
    require((flags & O_DIRECT) && (descriptor_flags & FD_CLOEXEC), "O_DIRECT/FD_CLOEXEC missing");
#if defined(STATX_DIOALIGN)
    struct statx sx = {};
    system_check(statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0, "statx STATX_DIOALIGN (required)");
    require((sx.stx_mask & STATX_DIOALIGN) && sx.stx_dio_mem_align && sx.stx_dio_offset_align,
            "filesystem does not report supported STATX_DIOALIGN; no fallback configured");
    if (mem_alignment_) {
        require(mem_alignment_ == sx.stx_dio_mem_align && io_alignment_ == sx.stx_dio_offset_align,
                "direct alignment changed between files");
    } else {
        mem_alignment_ = sx.stx_dio_mem_align;
        io_alignment_ = sx.stx_dio_offset_align;
    }
#else
    require(false, "build headers lack STATX_DIOALIGN; no qualified fallback configured");
#endif
    last_fd_flags_ = flags;
    last_fd_descriptor_flags_ = descriptor_flags;
}
int DiskBackend::create_slot_file_(DiskBackupSlot& slot, bool readonly_slot) {
    std::string name = dir_ + (readonly_slot ? "/tms_ro_XXXXXX" : "/tms_XXXXXX");
    std::vector<char> chars(name.begin(), name.end());
    chars.push_back('\0');
    int fd = mkostemp(chars.data(), O_DIRECT | O_CLOEXEC);
    system_check(fd >= 0, "mkostemp O_DIRECT");
    slot.path = chars.data();
    try { check_fd_(fd); }
    catch (...) { close(fd); release(slot); throw; }
    return fd;
}
int DiskBackend::open_slot_file_(const DiskBackupSlot& slot) {
    int fd = open(slot.path.c_str(), O_RDWR | O_DIRECT | O_CLOEXEC | O_NOFOLLOW);
    system_check(fd >= 0, "reopen O_DIRECT");
    try { check_fd_(fd); }
    catch (...) { close(fd); throw; }
    return fd;
}
void DiskBackend::verify_numa_() const {
#if defined(USE_CUDA) && defined(__linux__)
    verify_staging_pages_numa(
        staging_buf_, chunk_bytes_, page_bytes_, numa_node_,
        [](size_t count, void** pages, int* status) {
            return syscall(SYS_move_pages, 0, count, pages, nullptr, status, 0);
        });
#endif
}
void DiskBackend::ensure_staging_buf_() {
#if defined(USE_CUDA) && defined(__linux__)
    if (staging_buf_) return;
    long page = sysconf(_SC_PAGESIZE);
    require(page > 0, "cannot query page size");
    page_bytes_ = static_cast<size_t>(page);
    size_t alignment = std::lcm(page_bytes_, mem_alignment_);
    require(chunk_bytes_ > 0 && chunk_bytes_ <= std::numeric_limits<size_t>::max() - alignment,
            "staging size overflow");
    require(chunk_bytes_ % page_bytes_ == 0 && chunk_bytes_ % io_alignment_ == 0 &&
            chunk_bytes_ % mem_alignment_ == 0, "staging chunk must satisfy page and direct I/O alignments");
    mapping_bytes_ = round_up(chunk_bytes_ + alignment, page_bytes_);
    mapping_base_ = mmap(nullptr, mapping_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping_base_ == MAP_FAILED) { mapping_base_ = nullptr; system_check(false, "anonymous mmap staging"); }
    void* aligned = reinterpret_cast<void*>(round_up(reinterpret_cast<uintptr_t>(mapping_base_), alignment));
    try {
        const size_t bits = 8 * sizeof(unsigned long);
        std::vector<unsigned long> mask(numa_node_ / bits + 1, 0);
        mask[numa_node_ / bits] |= 1ul << (numa_node_ % bits);
        system_check(syscall(SYS_mbind, aligned, chunk_bytes_, MPOL_BIND, mask.data(), mask.size() * bits, 0) == 0,
                     "mbind MPOL_BIND before first touch");
#ifdef MADV_NOHUGEPAGE
        // Avoid THP collapse migration PTEs across the staging range on 64 KB-page systems.
        if (madvise(aligned, chunk_bytes_, MADV_NOHUGEPAGE) != 0) {}
#endif
        memset(aligned, 0, chunk_bytes_);
        if (mlock(aligned, chunk_bytes_) != 0) {}
        cudaError_t result = cudaHostRegister(aligned, chunk_bytes_, cudaHostRegisterPortable);
        require(result == cudaSuccess, std::string("cudaHostRegister: ") + cudaGetErrorString(result));
        staging_buf_ = aligned;
        verify_numa_();
    } catch (...) {
        if (staging_buf_) cudaHostUnregister(staging_buf_);
        staging_buf_ = nullptr;
        munmap(mapping_base_, mapping_bytes_);
        mapping_base_ = nullptr;
        throw;
    }
#else
    require(false, "NUMA-bound direct disk backup requires Linux CUDA");
#endif
}
void DiskBackend::configure(const std::string& dir, int numa_node) {
    require(numa_node >= 0 && numa_node < 1048576, "invalid NUMA node");
    std::string normalized = std::filesystem::canonical(dir).string();
    if (configured_) {
        require(dir_ == normalized && numa_node_ == numa_node, "conflicting process-global placement");
        return;
    }
    require(!allocation_started_, "configure before disk allocations");
    if (chunk_bytes_ == 0) chunk_bytes_ = compute_disk_chunk_bytes_from_env();
    require(std::filesystem::exists("/sys/devices/system/node/node" + std::to_string(numa_node)), "NUMA node does not exist");
    struct statfs fs;
    system_check(statfs(normalized.c_str(), &fs) == 0, "statfs");
    require(fs.f_type != 0x01021994 && fs.f_type != 0x858458f6, "tmpfs/ramfs cannot back disk offload");
    dir_ = normalized;
    numa_node_ = numa_node;
    DiskBackupSlot probe;
    try {
        File file(create_slot_file_(probe));
        ensure_staging_buf_();
        // Probe both directions through the actual registered staging buffer.
        size_t n = std::lcm(mem_alignment_, io_alignment_);
        require(n <= chunk_bytes_, "alignment exceeds staging size");
        memset(staging_buf_, 0xa5, n);
        direct_io(file.fd, staging_buf_, n, 0, io_alignment_, mem_alignment_, true);
        memset(staging_buf_, 0, n);
        direct_io(file.fd, staging_buf_, n, 0, io_alignment_, mem_alignment_, false);
        for (size_t i = 0; i < n; ++i) require(static_cast<uint8_t*>(staging_buf_)[i] == 0xa5, "direct probe data mismatch");
        file.finish();
        release(probe);
        configured_ = true;
    } catch (...) {
        release(probe);
#if defined(USE_CUDA) && defined(__linux__)
        if (staging_buf_) cudaHostUnregister(staging_buf_);
        if (mapping_base_) munmap(mapping_base_, mapping_bytes_);
#endif
        staging_buf_ = mapping_base_ = nullptr;
        mem_alignment_ = io_alignment_ = 0;
        throw;
    }
}
void DiskBackend::stream_(int fd, void* gpu_ptr, size_t size, bool to_disk) {
    verify_numa_();
    for (size_t offset = 0; offset < size;) {
        size_t n = std::min(chunk_bytes_, size - offset);
        size_t padded = round_up(n, io_alignment_);
        if (to_disk) {
            CUDA_ERROR_CHECK(cudaMemcpy(staging_buf_, static_cast<uint8_t*>(gpu_ptr) + offset, n, cudaMemcpyDeviceToHost));
            memset(static_cast<uint8_t*>(staging_buf_) + n, 0, padded - n);
            direct_io(fd, staging_buf_, padded, offset, io_alignment_, mem_alignment_, true);
            logical_write_bytes_ += n; padded_write_bytes_ += padded;
        } else {
            direct_io(fd, staging_buf_, padded, offset, io_alignment_, mem_alignment_, false);
            CUDA_ERROR_CHECK(cudaMemcpy(static_cast<uint8_t*>(gpu_ptr) + offset, staging_buf_, n, cudaMemcpyHostToDevice));
            logical_read_bytes_ += n; padded_read_bytes_ += padded;
        }
        offset += n;
    }
}
void DiskBackend::offload(void* gpu_ptr, size_t size, DiskBackupSlot& slot, bool readonly_slot) {
    try {
        require(configured_, "direct disk backend is not configured");
        File file(slot.path.empty() ? create_slot_file_(slot, readonly_slot) : open_slot_file_(slot));
        slot.logical_bytes = size;
        slot.padded_bytes = round_up(size, io_alignment_);
        require(slot.padded_bytes <= static_cast<size_t>(std::numeric_limits<off_t>::max()), "file size overflow");
        system_check(ftruncate(file.fd, slot.padded_bytes) == 0, "ftruncate padded file");
        stream_(file.fd, gpu_ptr, size, true);
        file.finish();
    } catch (const std::exception& e) { SIMPLE_CHECK(false, e.what()); }
}
void DiskBackend::onload(void* gpu_ptr, size_t size, DiskBackupSlot& slot) {
    try {
        require(configured_ && !slot.path.empty() && slot.logical_bytes == size, "missing/mismatched disk backup");
        File file(open_slot_file_(slot));
        struct stat st;
        system_check(fstat(file.fd, &st) == 0, "fstat backup");
        require(st.st_size >= 0 && static_cast<size_t>(st.st_size) == slot.padded_bytes, "backup padded length mismatch");
        stream_(file.fd, gpu_ptr, size, false);
        file.finish();
    } catch (const std::exception& e) { SIMPLE_CHECK(false, e.what()); }
}
void DiskBackend::release(DiskBackupSlot& slot) {
    if (!slot.path.empty()) {
        SIMPLE_CHECK(unlink(slot.path.c_str()) == 0 || errno == ENOENT, "failed to unlink disk backup");
        slot = DiskBackupSlot{};
    }
}
std::string DiskBackend::info() const {
    std::ostringstream out;
    out << "{\"direct\":" << (configured_ ? "true" : "false")
        << ",\"directory\":" << json_string(dir_)
        << ",\"staging_address\":" << reinterpret_cast<uintptr_t>(staging_buf_)
        << ",\"staging_bytes\":" << (staging_buf_ ? chunk_bytes_ : 0)
        << ",\"requested_numa_node\":" << numa_node_
        << ",\"observed_numa_nodes\":" << (staging_buf_ ? "[" + std::to_string(numa_node_) + "]" : "[]")
        << ",\"page_bytes\":" << page_bytes_
        << ",\"memory_alignment\":" << mem_alignment_ << ",\"io_alignment\":" << io_alignment_
        << ",\"alignment_source\":\"STATX_DIOALIGN\""
        << ",\"fd_flags\":" << last_fd_flags_ << ",\"fd_descriptor_flags\":" << last_fd_descriptor_flags_
        << ",\"logical_write_bytes\":" << logical_write_bytes_ << ",\"padded_write_bytes\":" << padded_write_bytes_
        << ",\"logical_read_bytes\":" << logical_read_bytes_ << ",\"padded_read_bytes\":" << padded_read_bytes_
        << ",\"readonly_skipped_write_bytes\":" << readonly_skipped_write_bytes_
        << ",\"readonly_skipped_pauses\":" << readonly_skipped_pauses_ << "}";
    return out.str();
}
