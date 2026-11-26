#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"
#if defined(USE_MUSA)
#include "hardware_musa_support.h"
#endif

#include <algorithm>
#include <set>

#if !TMS_ROCM_LEGACY_CHUNKED && !defined(USE_XPU)
namespace {
class ScopedCudaDevice {
public:
    explicit ScopedCudaDevice(CUdevice target_device)
        : target_(static_cast<int>(target_device)) {
#if defined(USE_CUDA)
        CURESULT_CHECK(cuCtxGetCurrent(&saved_ctx_));
        if (saved_ctx_ != nullptr) {
            CUdevice ctx_dev = 0;
            CURESULT_CHECK(cuCtxGetDevice(&ctx_dev));
            original_ = static_cast<int>(ctx_dev);
        } else {
            CUDA_ERROR_CHECK(cudaGetDevice(&original_));
        }
#else
        CUDA_ERROR_CHECK(cudaGetDevice(&original_));
#endif
        if (original_ != target_) {
            CUDA_ERROR_CHECK(cudaSetDevice(target_));
            switched_ = true;
        }
    }
    ~ScopedCudaDevice() {
        if (!switched_) {
            return;
        }
#if defined(USE_CUDA)
        if (saved_ctx_ != nullptr) {
            CURESULT_CHECK(cuCtxSetCurrent(saved_ctx_));
            return;
        }
#endif
        if (original_ >= 0) {
            CUDA_ERROR_CHECK(cudaSetDevice(original_));
        }
    }
    ScopedCudaDevice(const ScopedCudaDevice&) = delete;
    ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

private:
#if defined(USE_CUDA)
    CUcontext saved_ctx_ = nullptr;
#endif
    int original_ = -1;
    int target_ = -1;
    bool switched_ = false;
};
}  // namespace
#endif

TorchMemorySaver::TorchMemorySaver()
    : disk_backend_(compute_disk_backup_dir_from_env(), 0) {
#if TMS_SUPPORTS_RETAIN_CPU_BACKUP
    retain_cpu_backup_.store(get_bool_env_var("TMS_RETAIN_CPU_BACKUP"));
#endif
}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(
    void **ptr,
    CUdevice device,
    size_t raw_size,
    const std::string& tag,
    const bool enable_cpu_backup,
    const CpuBackupKind cpu_backup_kind,
    const bool enable_disk_backup,
    const bool write_once) {
    // Enforce here, not only in the Python layer: an assert is stripped under
    // python -O and bypassed by direct C-API / env-var use.
    SIMPLE_CHECK(!write_once || enable_cpu_backup || enable_disk_backup,
                 "write_once requires enable_cpu_backup or enable_disk_backup");
    SIMPLE_CHECK(!(enable_cpu_backup && enable_disk_backup),
                 "cpu_backup and disk_backup are mutually exclusive");
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(!enable_disk_backup, "disk backup is not supported on the ROCm 6.x legacy chunked path");
    SIMPLE_CHECK(!write_once, "write_once is not supported on the ROCm 6.x legacy chunked path");
    return ROCmHIPImplementation::rocm_malloc(
        ptr,
        device,
        raw_size,
        tag,
        enable_cpu_backup,
        cpu_backup_kind,
        allocation_metadata_,
        allocator_metadata_mutex_);

#elif defined(USE_XPU)
    SIMPLE_CHECK(!enable_disk_backup, "disk backup is not supported on Intel XPU");
    SIMPLE_CHECK(!write_once, "write_once is not supported on Intel XPU");
    return XPUImplementation::xpu_malloc(ptr, device, raw_size, tag, enable_cpu_backup, allocation_metadata_, allocator_metadata_mutex_);

#else
#if !defined(USE_CUDA) || !defined(__linux__)
    SIMPLE_CHECK(!enable_disk_backup, "direct disk backup requires Linux CUDA");
#endif
#if defined(USE_MUSA) || defined(USE_MLU)
    SIMPLE_CHECK(!write_once, "write_once is not supported on MUSA or MLU");
#endif
    if (enable_disk_backup) {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        disk_backend_.require_configured(static_cast<int>(device));
    }
    const size_t allocation_size = CUDAUtils::cu_mem_get_allocation_size(raw_size, device);
    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + allocation_size > free_bytes) {
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since"
                << " memory_margin_bytes=" << memory_margin_bytes
                << " allocation_size=" << allocation_size
                << " free_bytes=" << free_bytes
                << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    CUmemGenericAllocationHandle allocHandle;

    cudaError_t ret = CUDAUtils::cu_mem_create(&allocHandle, allocation_size, device);
    if (ret != cudaSuccess) {
        return ret;
    }

    CURESULT_CHECK(cuMemAddressReserve(
        (CUdeviceptr *) ptr, CUDAUtils::cu_mem_get_reserve_size(allocation_size, device), 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, allocation_size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, allocation_size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{
                raw_size, device, tag, AllocationState::ACTIVE,
                enable_cpu_backup, CpuBackupSlot{cpu_backup_kind, nullptr, 0},
                enable_disk_backup, DiskBackupSlot{}, allocation_size, allocHandle,
                write_once}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " raw_size=" << raw_size
              << " allocation_size=" << allocation_size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_XPU)
    return XPUImplementation::xpu_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#else
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    {
        ScopedCudaDevice dev_guard(metadata.device);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }

    if (AllocationState::ACTIVE == metadata.state) {
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.allocation_size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    }
    CURESULT_CHECK(cuMemAddressFree(
        (CUdeviceptr) ptr, CUDAUtils::cu_mem_get_reserve_size(metadata.allocation_size, metadata.device)));

    cpu_backuper::release(metadata.cpu_backup);

    if (metadata.enable_disk_backup) {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        disk_backend_.release(metadata.disk);
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free "
              << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
              << " metadata.allocation_size=" << metadata.allocation_size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

#if defined(USE_XPU)
uint64_t TorchMemorySaver::xpu_committed_bytes(int device_id) {
    return XPUImplementation::xpu_committed_bytes(
        device_id, allocation_metadata_, allocator_metadata_mutex_);
}

uint64_t TorchMemorySaver::xpu_leaked_bytes(int device_id) {
    return XPUImplementation::xpu_leaked_bytes(
        device_id, allocation_metadata_, allocator_metadata_mutex_);
}

uint64_t TorchMemorySaver::xpu_tracked_bytes(int device_id) {
    return XPUImplementation::xpu_tracked_bytes(
        device_id, allocation_metadata_, allocator_metadata_mutex_);
}
#endif

uint32_t TorchMemorySaver::affected_devices(const char* tag, int* out_device_ids, uint32_t capacity) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    const bool all = (tag == nullptr || tag[0] == '\0');
    std::set<int> devices;
    for (const auto &kv : allocation_metadata_) {
        const AllocationMetadata &metadata = kv.second;
        if (!all && metadata.tag != tag)
            continue;
        devices.insert(static_cast<int>(metadata.device));
    }
    const auto count = static_cast<uint32_t>(devices.size());
    if (out_device_ids != nullptr) {
        std::copy_n(devices.begin(), std::min(capacity, count), out_device_ids);
    }
    return count;
}

cudaError_t TorchMemorySaver::pause(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);
    return cudaSuccess;

#elif defined(USE_XPU)
    return XPUImplementation::xpu_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::ACTIVE) {
            std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        if (metadata.enable_cpu_backup) {
            if (metadata.write_once && metadata.cpu_backup.data != nullptr) {
                SIMPLE_CHECK(metadata.cpu_backup.size == metadata.raw_size,
                             "write_once cpu_backup slot size mismatch on skip");
            } else {
                ScopedCudaDevice dev_guard(metadata.device);
                cpu_backuper::offload(ptr, metadata.raw_size, metadata.cpu_backup);
            }
        } else if (metadata.enable_disk_backup) {
            if (metadata.write_once && metadata.disk.populated) {
                SIMPLE_CHECK(!metadata.disk.path.empty() && metadata.disk.logical_bytes == metadata.raw_size,
                             "write_once disk slot state mismatch on skip");
                disk_backend_.record_readonly_skip(metadata.raw_size);
            } else {
                ScopedCudaDevice dev_guard(metadata.device);
                disk_backend_.offload(ptr, metadata.raw_size, metadata.disk);
            }
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.allocation_size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
                  << " metadata.allocation_size=" << metadata.allocation_size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
    return cudaSuccess;
#endif
}

cudaError_t TorchMemorySaver::resume(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);
    return cudaSuccess;

#elif defined(USE_XPU)
    return XPUImplementation::xpu_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
    const bool retain_cpu_backup = retain_cpu_backup_.load();

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        CUmemGenericAllocationHandle newAllocHandle;
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(
            &newAllocHandle, metadata.allocation_size, metadata.device));

        CURESULT_CHECK(cuMemMap(
            (CUdeviceptr) ptr, metadata.allocation_size, 0, newAllocHandle, 0));

        CUDAUtils::cu_mem_set_access(ptr, metadata.allocation_size, metadata.device);

#if defined(USE_MUSA)
        // MUSA may hand back non-zero pages for a fresh mapping, so zero a
        // region that has no backup to restore. Backed regions are overwritten
        // by onload, and zeroing them first would double the resume traffic.
        if (!metadata.enable_cpu_backup && !metadata.enable_disk_backup) {
            MUSAImplementation::zero_region(
                ptr, metadata.allocation_size, static_cast<int>(metadata.device));
        }
#endif

        if (metadata.enable_cpu_backup) {
            {
                ScopedCudaDevice dev_guard(metadata.device);
                cpu_backuper::onload(
                    ptr, metadata.raw_size, metadata.device, metadata.cpu_backup);
            }
            if (!retain_cpu_backup && !metadata.write_once) {
                cpu_backuper::release(metadata.cpu_backup);
            }
        } else if (metadata.enable_disk_backup) {
            {
                ScopedCudaDevice dev_guard(metadata.device);
                disk_backend_.onload(ptr, metadata.raw_size, metadata.disk);
            }
            if (!metadata.write_once) {
                disk_backend_.release(metadata.disk);
            }
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
                  << " metadata.allocation_size=" << metadata.allocation_size
                  << " (old)metadata.allocHandle=" << metadata.allocHandle
                  << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }
    return cudaSuccess;
#endif
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED || defined(USE_XPU)
        size_t total_size = metadata.aligned_size;
#else
        size_t total_size = metadata.raw_size;
#endif

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + total_size)) {
            const size_t offset = query_gpu_ptr - ptr;
            // Disk-backed allocations have no CPU-resident copy; callers must use the resumed GPU tensor.
            if (metadata.enable_disk_backup) {
                return nullptr;
            }
            if (metadata.state == AllocationState::ACTIVE) {
                return metadata.cpu_backup.data == nullptr
                    ? nullptr
                    : (uint8_t*) metadata.cpu_backup.data + offset;
            } else {
                SIMPLE_CHECK(nullptr != metadata.cpu_backup.data,
                    "get_cpu_backup_pointer: found paused allocation but cpu_backup does not exist, do you forget to enable cpu backup");
                return (uint8_t*) metadata.cpu_backup.data + offset;
            }
        }
    }

    std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer fail to find backup "
              << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
              << std::endl;
    exit(1);
}
