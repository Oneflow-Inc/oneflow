#include "tensorflow/stream_executor/host/host_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/engine/xla/of2xla/memory/device_memory_pool.h"

namespace oneflow {
namespace mola {
namespace memory {

class CpuMemoryPool : public DeviceMemoryPool {
 public:
  explicit CpuMemoryPool(se::Stream *stream, int device_ordinal)
      : DeviceMemoryPool(stream, device_ordinal) {}
  virtual ~CpuMemoryPool() { Release(); }

 private:
  void ReserveImpl(size_t size) override {
    mem_buffer_ = new uint8_t[size];
    CHECK(mem_buffer_);
    capacity_ = size;
  }

  void ReleaseImpl() override {
    if (capacity_ > 0 && mem_buffer_) {
      delete[] mem_buffer_;
    }
    capacity_ = 0;
    mem_buffer_ = nullptr;
  }
};

REGISTER_XLA_MEMORY_POOL(se::host::kHostPlatformId, CpuMemoryPool);

class GpuMemoryPool : public DeviceMemoryPool {
 public:
  explicit GpuMemoryPool(se::Stream *stream, int device_ordinal)
      : DeviceMemoryPool(stream, device_ordinal) {}

  virtual ~GpuMemoryPool() { Release(); }

 private:
  void ReserveImpl(size_t size) override {
#ifdef WITH_CUDA
    int device_ordinal;
    cudaGetDevice(&device_ordinal);
    if (device_ordinal != device_ordinal_) {
      cudaSetDevice(device_ordinal_);
    }

    CudaCheck(cudaMalloc(&mem_buffer_, size));

    if (device_ordinal != device_ordinal_) {
      cudaSetDevice(device_ordinal);
    }
#else
    LOG(FATAL) << "Recompile with CUDA.";
#endif
    CHECK(mem_buffer_);
    capacity_ = size;
  }

  void ReleaseImpl() override {
#ifdef WITH_CUDA
    int device_ordinal;
    cudaGetDevice(&device_ordinal);
    if (device_ordinal != device_ordinal_) {
      cudaSetDevice(device_ordinal_);
    }

    if (capacity_ > 0 && mem_buffer_) {
      CudaCheck(cudaFree(mem_buffer_));
    }

    if (device_ordinal != device_ordinal_) {
      cudaSetDevice(device_ordinal);
    }
#else
    LOG(FATAL) << "Recompile with CUDA.";
#endif
    capacity_ = 0;
    mem_buffer_ = nullptr;
  }
};

REGISTER_XLA_MEMORY_POOL(se::cuda::kCudaPlatformId, GpuMemoryPool);

}  // namespace memory

void DeviceMemoryPool::Reserve(size_t size) {
  if (limited_memory_size_ > -1) {
    CHECK_LT(size, limited_memory_size_);
  }
  
  while (size > capacity_) {
    // Block host to ensure that all the launched kernels depend on this
    // memory buffer have been executed completely
    CHECK(stream_->BlockHostUntilDone().ok());
  
    ReleaseImpl();

    ReserveImpl(size);
  }
}

void DeviceMemoryPool::Release() {
  // Block host to ensure that all the launched kernels depend on this
  // memory buffer have been executed completely
  CHECK(stream_->BlockHostUntilDone().ok());

  ReleaseImpl();
}

typedef DeviceMemoryPool::MemPoolFactory MemPoolFactory;
typedef std::unordered_map<se::Platform::Id, MemPoolFactory> MemPoolFactoryMap;

static MemPoolFactoryMap* GlobalMemPoolFactories() {
  static MemPoolFactoryMap factories;
  return &factories;
}

/*static*/ std::shared_ptr<DeviceMemoryPool> DeviceMemoryPool::NewMemoryPool(
    const se::Platform *platform, se::Stream *stream, int device_ordinal) {
  MemPoolFactoryMap *factories = GlobalMemPoolFactories();
  const auto &it = factories->find(platform->id());
  CHECK(it != factories->end())
      << "DeviceMemoryPool has not been registered for platform id "
      << platform->id();
  DeviceMemoryPool *mem_pool = (it->second)(stream, device_ordinal);
  return std::shared_ptr<DeviceMemoryPool>(mem_pool);
}

/*static*/ void DeviceMemoryPool::RegisterFactory(
    const se::Platform::Id &platform_id, MemPoolFactory factory) {
  MemPoolFactoryMap *factories = GlobalMemPoolFactories();
  if (factories->count(platform_id)) {
    DLOG(WARNING) << "DeviceMemoryPool for platform id (" << platform_id
                  << ") has been registed more than once";
  }
  factories->emplace(platform_id, factory);
}

}  // namespace mola
}  // namespace oneflow
