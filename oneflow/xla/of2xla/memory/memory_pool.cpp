#include "oneflow/core/device/cuda_util.h"
#include "oneflow/xla/of2xla/memory/memory_pool.h"

namespace oneflow {
namespace mola {
namespace memory {

class CpuMemoryPool : public DeviceMemoryPool {
 public:
  explicit CpuMemoryPool(int device_ordinal)
      : DeviceMemoryPool(device_ordinal) {}
  virtual ~CpuMemoryPool() { Release(); }

  void Reserve(size_t size) override {
    if (limited_memory_size_ > -1) {
      CHECK_LT(size, limited_memory_size_);
    }
    if (size > capacity_) {
      Release();
      mem_buffer_ = new uint8_t[size];
      capacity_ = size;
    }
  }

  void Release() override {
    if (capacity_ > 0 && mem_buffer_) {
      delete[] mem_buffer_;
    }
    capacity_ = 0;
    mem_buffer_ = nullptr;
  }
};

class GpuMemoryPool : public DeviceMemoryPool {
 public:
  explicit GpuMemoryPool(const void *cuda_stream, int device_ordinal)
      : DeviceMemoryPool(device_ordinal), cuda_stream_(cuda_stream) {}

  virtual ~GpuMemoryPool() { Release(); }

  void Reserve(size_t size) override {
    if (limited_memory_size_ > -1) {
      CHECK_LT(size, limited_memory_size_);
    }
    if (size > capacity_) {
      Release();
#ifdef WITH_CUDA
      CudaCheck(cudaMalloc(&mem_buffer_, size));
#else
      LOG(FATAL) << "Recompile with CUDA.";
#endif
      CHECK_NOTNULL(mem_buffer_);
      capacity_ = size;
    }
  }

  void Release() override {
#ifdef WITH_CUDA
    int device_ordinal;
    cudaGetDevice(&device_ordinal);
    if (device_ordinal != device_ordinal_) {
      cudaSetDevice(device_ordinal_);
    }

    // Synchronize cuda stream to ensure that all the launched kernel depend
    // on this memory buffer have been executed completely
    cudaStream_t stream = reinterpret_cast<CUstream_st *>(
        const_cast<void *>(cuda_stream_));
    CudaCheck(cudaStreamSynchronize(stream));
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

 private:
  const void *cuda_stream_;
};

}  // namespace memory

/*static*/ DeviceMemoryPool *DeviceMemoryPool::NewCpuMemoryPool(
    int device_ordinal) {
  return new memory::CpuMemoryPool(device_ordinal);
}

/*static*/ DeviceMemoryPool *DeviceMemoryPool::NewGpuMemoryPool(
    const void *cuda_stream, int device_ordinal) {
  return new memory::GpuMemoryPool(cuda_stream, device_ordinal);
}

}  // namespace mola
}  // namespace oneflow
