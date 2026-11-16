#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

struct DiskBackupSlot {
    std::string path;
    size_t logical_bytes = 0;
    size_t padded_bytes = 0;
};

// Synchronous disk transfers share one bounded, NUMA-bound pinned buffer.
// All methods are serialized by the allocator metadata mutex.
class DiskBackend {
public:
    DiskBackend(std::string dir, size_t chunk_bytes);
    ~DiskBackend();
    void set_dir(const std::string& dir);
    void configure(const std::string& dir, int numa_node);
    void require_configured();
    std::string info() const;
    void offload(void* gpu_ptr, size_t size, DiskBackupSlot& slot);
    void onload(void* gpu_ptr, size_t size, DiskBackupSlot& slot);
    void release(DiskBackupSlot& slot);

private:
    void ensure_staging_buf_();
    void verify_numa_() const;
    int create_slot_file_(DiskBackupSlot& slot);
    int open_slot_file_(const DiskBackupSlot& slot);
    void check_fd_(int fd);
    void stream_(int fd, void* gpu_ptr, size_t size, bool to_disk);

    std::string dir_;
    size_t chunk_bytes_;
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
    bool allocation_started_ = false;
    uint64_t logical_write_bytes_ = 0, padded_write_bytes_ = 0;
    uint64_t logical_read_bytes_ = 0, padded_read_bytes_ = 0;
};

std::string compute_disk_backup_dir_from_env();
size_t compute_disk_chunk_bytes_from_env();
