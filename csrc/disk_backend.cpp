#include "disk_backend.h"
#include "macro.h"
#include "utils.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
#if defined(__linux__)
#ifndef AT_EMPTY_PATH
#define AT_EMPTY_PATH 0x1000
#endif
#ifndef SYS_statx
#if defined(__x86_64__)
#define SYS_statx 332
#elif defined(__aarch64__)
#define SYS_statx 291
#endif
#endif

constexpr uint32_t kTmsStatxDioAlign = 0x00002000U;

struct TmsStatxTimestamp {
    int64_t tv_sec;
    uint32_t tv_nsec;
    int32_t __reserved;
};

struct TmsStatx {
    uint32_t stx_mask;
    uint32_t stx_blksize;
    uint64_t stx_attributes;
    uint32_t stx_nlink;
    uint32_t stx_uid;
    uint32_t stx_gid;
    uint16_t stx_mode;
    uint16_t __spare0[1];
    uint64_t stx_ino;
    uint64_t stx_size;
    uint64_t stx_blocks;
    uint64_t stx_attributes_mask;
    TmsStatxTimestamp stx_atime;
    TmsStatxTimestamp stx_btime;
    TmsStatxTimestamp stx_ctime;
    TmsStatxTimestamp stx_mtime;
    uint32_t stx_rdev_major;
    uint32_t stx_rdev_minor;
    uint32_t stx_dev_major;
    uint32_t stx_dev_minor;
    uint64_t stx_mnt_id;
    uint32_t stx_dio_mem_align;
    uint32_t stx_dio_offset_align;
    uint64_t stx_subvol;
    uint32_t stx_atomic_write_unit_min;
    uint32_t stx_atomic_write_unit_max;
    uint32_t stx_atomic_write_segments_max;
    uint32_t __spare1[1];
    uint64_t __spare3[9];
};

static_assert(sizeof(TmsStatx) == 0x100, "TmsStatx size mismatch");
static_assert(offsetof(TmsStatx, stx_dio_mem_align) == 0x98, "stx_dio_mem_align offset mismatch");
static_assert(offsetof(TmsStatx, stx_dio_offset_align) == 0x9c, "stx_dio_offset_align offset mismatch");

bool query_fd_dioalign(int fd, uint32_t& out_mask, uint32_t& out_mem_align, uint32_t& out_offset_align) {
#if defined(SYS_statx)
    TmsStatx sx = {};
    if (syscall(SYS_statx, fd, "", AT_EMPTY_PATH, kTmsStatxDioAlign, &sx) != 0) {
        return false;
    }
    out_mask = sx.stx_mask;
    out_mem_align = sx.stx_dio_mem_align;
    out_offset_align = sx.stx_dio_offset_align;
    return true;
#else
    (void)fd;
    (void)out_mask;
    (void)out_mem_align;
    (void)out_offset_align;
    errno = ENOSYS;
    return false;
#endif
}
#endif

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
    static constexpr char kHex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20) {
            out += "\\u00";
            out += kHex[c >> 4];
            out += kHex[c & 15];
        } else {
            out += static_cast<char>(c);
        }
    }
    return out + '"';
}

#if defined(USE_CUDA) && defined(__linux__)
int detect_cuda_device_numa_node(int cuda_device) {
    int device = cuda_device;
    if (device < 0) {
        cudaError_t dev_err = cudaGetDevice(&device);
        require(dev_err == cudaSuccess, std::string("cudaGetDevice: ") + cudaGetErrorString(dev_err));
    }
    char bus_id[64] = {};
    cudaError_t err = cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device);
    require(err == cudaSuccess, std::string("cudaDeviceGetPCIBusId: ") + cudaGetErrorString(err));
    std::string bdf(bus_id);
    for (char& c : bdf) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    // Normalize 8-digit PCI domain ("00000000:bb:dd.f") to Linux sysfs 4-digit ("0000:bb:dd.f").
    if (bdf.size() > 12 && bdf.rfind("0000", 0) == 0 && bdf[8] == ':') {
        bdf.erase(0, 4);
    }
    std::ifstream f("/sys/bus/pci/devices/" + bdf + "/numa_node");
    require(f.is_open(), "cannot open NUMA sysfs for PCI device " + bdf);
    int node = -1;
    f >> node;
    require(!f.fail(), "cannot parse NUMA sysfs for PCI device " + bdf);
    // Linux reports -1 on single-socket or virtualized hosts without NUMA affinity.
    return node < 0 ? 0 : node;
}

void reject_tmpfs_before_mkdir(const std::string& dir) {
    std::error_code ec;
    std::filesystem::path probe = std::filesystem::absolute(dir, ec);
    while (!ec && !probe.empty() && !std::filesystem::exists(probe, ec)) {
        std::filesystem::path parent = probe.parent_path();
        if (parent == probe) break;
        probe = std::move(parent);
    }
    if (!ec && std::filesystem::exists(probe, ec)) {
        struct statfs fs{};
        if (statfs(probe.c_str(), &fs) == 0) {
            require(fs.f_type != 0x01021994 && fs.f_type != 0x858458f6,
                    "tmpfs/ramfs cannot back disk offload");
        }
    }
}
#endif

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
    return (dir && dir[0] != '\0') ? std::string(dir) : std::string();
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

bool DiskBackend::owns_process_resources_() const noexcept {
    return owner_pid_ != 0 && ::getpid() == owner_pid_;
}

bool DiskBackend::matches_configured_dir_(const std::string& dir) const noexcept {
    std::error_code ec;
    std::filesystem::path normalized = std::filesystem::canonical(dir, ec);
    return !ec && normalized.string() == dir_;
}

DiskBackend::~DiskBackend() {
#if defined(USE_CUDA) && defined(__linux__)
    if (owns_process_resources_() && mapping_base_ != nullptr) {
        if (staging_buf_ != nullptr) cudaHostUnregister(staging_buf_);
        munmap(mapping_base_, mapping_bytes_);
    }
#endif
    if (owns_process_resources_() && !proc_dir_.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(proc_dir_, ec);
    }
    proc_dir_.clear();
    owner_pid_ = 0;
    staging_buf_ = nullptr;
    mapping_base_ = nullptr;
    mapping_bytes_ = 0;
}

void DiskBackend::set_dir(const std::string& dir) {
    require(!dir.empty(), "disk backup directory must be nonempty");
    if (configured_) {
        require(matches_configured_dir_(dir), "placement is immutable after configuration");
        return;
    }
    dir_ = dir;
}

void DiskBackend::require_configured(int cuda_device) {
    try {
        if (!configured_) {
            require(!dir_.empty(),
                    "configure disk backup first: call configure_direct_disk_backup(path), "
                    "set_disk_backup_dir(path), or set TMS_DISK_BACKUP_DIR");
            configure(dir_, kAutoNumaNode, cuda_device);
        }
    } catch (const std::exception& e) {
        SIMPLE_CHECK(false, e.what());
    }
}

void DiskBackend::check_fd_(int fd) {
    int flags = fcntl(fd, F_GETFL);
    int descriptor_flags = fcntl(fd, F_GETFD);
    system_check(flags >= 0 && descriptor_flags >= 0, "fcntl");
    require((flags & O_DIRECT) && (descriptor_flags & FD_CLOEXEC), "O_DIRECT/FD_CLOEXEC missing");
#if defined(__linux__)
    uint32_t stx_mask = 0;
    uint32_t dio_mem_align = 0;
    uint32_t dio_offset_align = 0;
    // Call SYS_statx directly to avoid a GLIBC_2.28 statx symbol dependency in manylinux2014 wheels.
    system_check(query_fd_dioalign(fd, stx_mask, dio_mem_align, dio_offset_align),
                 "statx STATX_DIOALIGN (required)");
    require((stx_mask & kTmsStatxDioAlign) && dio_mem_align && dio_offset_align,
            "filesystem does not report supported STATX_DIOALIGN; no fallback configured");
    if (mem_alignment_) {
        require(mem_alignment_ == dio_mem_align && io_alignment_ == dio_offset_align,
                "direct alignment changed between files");
    } else {
        mem_alignment_ = dio_mem_align;
        io_alignment_ = dio_offset_align;
    }
#else
    require(false, "STATX_DIOALIGN requires Linux");
#endif
    last_fd_flags_ = flags;
    last_fd_descriptor_flags_ = descriptor_flags;
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
    require(owns_process_resources_(), "inherited pre-fork disk backend cannot allocate staging buffer in child");
    if (mapping_base_ != nullptr && staging_buf_ != nullptr) return;
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
        // Prevent concurrent fork() from marking O_DIRECT staging pages copy-on-write.
        system_check(madvise(mapping_base_, mapping_bytes_, MADV_DONTFORK) == 0,
                     "madvise MADV_DONTFORK staging");
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
        mapping_bytes_ = 0;
        throw;
    }
#else
    require(false, "NUMA-bound direct disk backup requires Linux CUDA");
#endif
}

void DiskBackend::configure(
    const std::string& dir,
    int numa_node,
    int cuda_device) {
    require(!dir.empty(), "disk backup directory must be nonempty");
    require(numa_node == kAutoNumaNode || (numa_node >= 0 && numa_node < 1048576), "invalid NUMA node");
#if !defined(USE_CUDA) || !defined(__linux__)
    (void)cuda_device;
    require(false, "NUMA-bound direct disk backup requires Linux CUDA");
#else
    int resolved_numa = numa_node == kAutoNumaNode ? detect_cuda_device_numa_node(cuda_device) : numa_node;
    require(resolved_numa >= 0 && resolved_numa < 1048576, "invalid NUMA node");
    if (configured_) {
        require(matches_configured_dir_(dir) && numa_node_ == resolved_numa,
                "conflicting process-global placement");
        return;
    }
    const size_t prev_chunk_bytes = chunk_bytes_;
    const size_t resolved_chunk_bytes = chunk_bytes_ ? chunk_bytes_ : compute_disk_chunk_bytes_from_env();
    require(std::filesystem::exists("/sys/devices/system/node/node" + std::to_string(resolved_numa)),
            "NUMA node does not exist");
    reject_tmpfs_before_mkdir(dir);
    std::filesystem::create_directories(dir);
    std::string normalized = std::filesystem::canonical(dir).string();
    struct stat path_st{};
    system_check(::lstat(normalized.c_str(), &path_st) == 0, "stat disk backup directory");
    require(S_ISDIR(path_st.st_mode), "disk backup directory must be a directory");
    struct statfs fs{};
    system_check(statfs(normalized.c_str(), &fs) == 0, "statfs");
    require(fs.f_type != 0x01021994 && fs.f_type != 0x858458f6, "tmpfs/ramfs cannot back disk offload");

    std::string prev_dir = dir_;
    const int prev_numa = numa_node_;
    const pid_t self_pid = ::getpid();
    std::string proc_dir = normalized + "/pid_" + std::to_string(static_cast<long long>(self_pid));
    std::string probe_path = proc_dir + "/probe.bin";
    int probe_fd = -1;
    try {
        std::error_code ec;
        std::filesystem::remove_all(proc_dir, ec);
        std::filesystem::create_directories(proc_dir);

        dir_ = std::move(normalized);
        proc_dir_ = proc_dir;
        owner_pid_ = self_pid;
        numa_node_ = resolved_numa;
        chunk_bytes_ = resolved_chunk_bytes;

        probe_fd = ::open(probe_path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0600);
        system_check(probe_fd >= 0, "open O_DIRECT probe");
        check_fd_(probe_fd);
        ensure_staging_buf_();

        // Probe both directions through the actual registered staging buffer.
        size_t n = std::lcm(mem_alignment_, io_alignment_);
        require(n <= chunk_bytes_, "alignment exceeds staging size");
        memset(staging_buf_, 0xa5, n);
        direct_io(probe_fd, staging_buf_, n, 0, io_alignment_, mem_alignment_, true);
        memset(staging_buf_, 0, n);
        direct_io(probe_fd, staging_buf_, n, 0, io_alignment_, mem_alignment_, false);
        for (size_t i = 0; i < n; ++i) {
            require(static_cast<uint8_t*>(staging_buf_)[i] == 0xa5, "direct probe data mismatch");
        }
        int close_rc = ::close(probe_fd);
        probe_fd = -1;
        system_check(close_rc == 0, "close direct probe");
        (void)::unlink(probe_path.c_str());
        configured_ = true;
    } catch (...) {
        if (probe_fd >= 0) {
            (void)::close(probe_fd);
        }
        (void)::unlink(probe_path.c_str());
        if (mapping_base_ != nullptr) {
            if (staging_buf_ != nullptr) cudaHostUnregister(staging_buf_);
            munmap(mapping_base_, mapping_bytes_);
        }
        staging_buf_ = mapping_base_ = nullptr;
        mapping_bytes_ = 0;
        page_bytes_ = 0;
        mem_alignment_ = io_alignment_ = 0;
        std::error_code ec;
        std::filesystem::remove_all(proc_dir, ec);
        proc_dir_.clear();
        owner_pid_ = 0;
        dir_ = std::move(prev_dir);
        numa_node_ = prev_numa;
        chunk_bytes_ = prev_chunk_bytes;
        throw;
    }
#endif
}

void DiskBackend::stream_(int fd, void* gpu_ptr, size_t size, bool to_disk) {
    ensure_staging_buf_();
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

void DiskBackend::offload(void* gpu_ptr, size_t size, DiskBackupSlot& slot) {
    try {
        require(configured_, "direct disk backend is not configured");
        require(owns_process_resources_(), "inherited pre-fork disk backend cannot be used in child process");
        size_t padded = round_up(size, io_alignment_);
        require(padded <= static_cast<size_t>(std::numeric_limits<off_t>::max()), "file size overflow");

        const bool was_populated = slot.populated;
        const size_t prev_padded = slot.padded_bytes;
        const bool assigned_new_path = slot.path.empty();
        if (assigned_new_path) {
            slot.path = proc_dir_ + "/slot_" + std::to_string(++next_slot_id_) + ".bin";
        }

        int fd = ::open(slot.path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0600);
        system_check(fd >= 0, "open O_DIRECT slot for write");
        try {
            check_fd_(fd);
            system_check(::ftruncate(fd, static_cast<off_t>(padded)) == 0, "ftruncate padded file");
            stream_(fd, gpu_ptr, size, true);
            int close_rc = ::close(fd);
            fd = -1;
            system_check(close_rc == 0, "close slot after write");
        } catch (...) {
            if (fd >= 0) {
                (void)::close(fd);
            }
            if (!was_populated) {
                (void)::unlink(slot.path.c_str());
                if (assigned_new_path) {
                    slot.path.clear();
                }
            }
            throw;
        }

        if (!was_populated) {
            ++active_slots_;
        }
        active_padded_bytes_ = active_padded_bytes_ - prev_padded + padded;
        slot.logical_bytes = size;
        slot.padded_bytes = padded;
        slot.populated = true;
    } catch (const std::exception& e) {
        SIMPLE_CHECK(false, e.what());
    }
}

void DiskBackend::onload(void* gpu_ptr, size_t size, DiskBackupSlot& slot) {
    try {
        require(configured_ && owns_process_resources_() && slot.populated &&
                    !slot.path.empty() && slot.logical_bytes == size,
                "missing/mismatched disk backup");
        int fd = ::open(slot.path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        system_check(fd >= 0, "open O_DIRECT slot for read");
        try {
            struct stat st{};
            system_check(::fstat(fd, &st) == 0, "fstat backup");
            require(st.st_size >= 0 && static_cast<size_t>(st.st_size) == slot.padded_bytes,
                    "backup padded length mismatch");
            stream_(fd, gpu_ptr, size, false);
            int close_rc = ::close(fd);
            fd = -1;
            system_check(close_rc == 0, "close slot after read");
        } catch (...) {
            if (fd >= 0) {
                (void)::close(fd);
            }
            throw;
        }
    } catch (const std::exception& e) {
        SIMPLE_CHECK(false, e.what());
    }
}

void DiskBackend::release(DiskBackupSlot& slot) {
    if (!slot.path.empty()) {
        if (owns_process_resources_()) {
            (void)::unlink(slot.path.c_str());
        }
        if (slot.populated) {
            active_padded_bytes_ = (active_padded_bytes_ >= slot.padded_bytes)
                ? (active_padded_bytes_ - slot.padded_bytes)
                : 0;
            if (active_slots_ > 0) {
                --active_slots_;
            }
        }
        slot.path.clear();
        slot.logical_bytes = 0;
        slot.padded_bytes = 0;
        slot.populated = false;
    }
}

std::string DiskBackend::info() const {
    const bool has_staging = owns_process_resources_() && staging_buf_ != nullptr;
    std::ostringstream out;
    out << "{\"direct\":" << (configured_ ? "true" : "false")
        << ",\"directory\":" << json_string(dir_)
        << ",\"process_directory\":" << json_string(proc_dir_)
        << ",\"staging_address\":" << (has_staging ? reinterpret_cast<uintptr_t>(staging_buf_) : 0)
        << ",\"staging_bytes\":" << (has_staging ? chunk_bytes_ : 0)
        << ",\"requested_numa_node\":" << numa_node_
        << ",\"observed_numa_nodes\":" << (has_staging ? "[" + std::to_string(numa_node_) + "]" : "[]")
        << ",\"page_bytes\":" << page_bytes_
        << ",\"memory_alignment\":" << mem_alignment_ << ",\"io_alignment\":" << io_alignment_
        << ",\"alignment_source\":\"STATX_DIOALIGN\""
        << ",\"fd_flags\":" << last_fd_flags_ << ",\"fd_descriptor_flags\":" << last_fd_descriptor_flags_
        << ",\"active_slots\":" << active_slots_ << ",\"active_padded_bytes\":" << active_padded_bytes_
        << ",\"logical_write_bytes\":" << logical_write_bytes_ << ",\"padded_write_bytes\":" << padded_write_bytes_
        << ",\"logical_read_bytes\":" << logical_read_bytes_ << ",\"padded_read_bytes\":" << padded_read_bytes_
        << ",\"readonly_skipped_write_bytes\":" << readonly_skipped_write_bytes_
        << ",\"readonly_skipped_pauses\":" << readonly_skipped_pauses_ << "}";
    return out.str();
}
