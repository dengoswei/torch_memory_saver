#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include "utils.h"

#ifndef TMS_HIDDEN
#if defined(__GNUC__) || defined(__clang__)
#define TMS_HIDDEN __attribute__((visibility("hidden")))
#else
#define TMS_HIDDEN
#endif
#endif

struct TMS_HIDDEN DiskBackupSlot {
    std::string path;
    size_t logical_bytes = 0;
    size_t padded_bytes = 0;
    bool populated = false;
};

// Synchronous O_DIRECT disk transfers through one NUMA-bound pinned staging buffer.
// All methods are serialized by the allocator metadata mutex.
class TMS_HIDDEN DiskBackend {
public:
    static constexpr int kAutoNumaNode = -1;

    DiskBackend(std::string dir, size_t chunk_bytes);
    ~DiskBackend();
    DiskBackend(const DiskBackend&) = delete;
    DiskBackend& operator=(const DiskBackend&) = delete;

    void set_dir(const std::string& dir);
    void configure(
        const std::string& dir,
        int numa_node = kAutoNumaNode,
        int cuda_device = -1);
    void require_configured(int cuda_device = -1);
    std::string info() const;
    void offload(void* gpu_ptr, size_t size, DiskBackupSlot& slot);
    void onload(void* gpu_ptr, size_t size, DiskBackupSlot& slot);
    void release(DiskBackupSlot& slot);
    void record_readonly_skip(size_t logical_bytes) {
        readonly_skipped_write_bytes_ += logical_bytes;
        ++readonly_skipped_pauses_;
    }

private:
    bool matches_configured_dir_(const std::string& dir) const noexcept;
    bool owns_process_resources_() const noexcept;
    void ensure_staging_buf_();
    void verify_numa_() const;
    void check_fd_(int fd);
    void stream_(int fd, void* gpu_ptr, size_t size, bool to_disk);

    std::string dir_;
    std::string proc_dir_;
    size_t chunk_bytes_;
    pid_t owner_pid_ = 0;
    uint64_t next_slot_id_ = 0;
    void* staging_buf_ = nullptr;
    void* mapping_base_ = nullptr;
    size_t mapping_bytes_ = 0;
    size_t page_bytes_ = 0;
    size_t mem_alignment_ = 0;
    size_t io_alignment_ = 0;
    int numa_node_ = -1;
    int last_fd_flags_ = 0;
    int last_fd_descriptor_flags_ = 0;
    bool configured_ = false;
    uint64_t active_slots_ = 0, active_padded_bytes_ = 0;
    uint64_t logical_write_bytes_ = 0, padded_write_bytes_ = 0;
    uint64_t logical_read_bytes_ = 0, padded_read_bytes_ = 0;
    uint64_t readonly_skipped_write_bytes_ = 0, readonly_skipped_pauses_ = 0;
};

TMS_HIDDEN std::string compute_disk_backup_dir_from_env();
TMS_HIDDEN size_t compute_disk_chunk_bytes_from_env();
