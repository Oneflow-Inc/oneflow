#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/compiler/of2xla/xla_allocator.h"

namespace oneflow {
namespace mola {

class GPUAllocator : public XlaAllocator {
 public:
  explicit GPUAllocator(const se::Platform* platform, int device_ordinal)
      : XlaAllocator(platform), device_ordinal_(device_ordinal) {}

  void* AllocateRaw(size_t alignment, size_t num_bytes) const override;

  void DeallocateRaw(se::DeviceMemoryBase mem) const override;

 private:
  int device_ordinal_;
};

REGISTER_XLA_ALLOCATOR(se::cuda::kCudaPlatformId, GPUAllocator);

void *GPUAllocator::AllocateRaw(size_t alignment, size_t num_bytes) const {
  void *p = nullptr;
#ifdef WITH_CUDA
  CudaCheck(cudaSetDevice(device_ordinal_));
  CudaCheck(cudaMalloc(&p, num_bytes));
#endif
  return p;
}

void GPUAllocator::DeallocateRaw(se::DeviceMemoryBase mem) const {
#ifdef WITH_CUDA
  CudaCheck(cudaSetDevice(device_ordinal_));
  CudaCheck(cudaFree(mem.opaque()));
#endif
}

}  // namespace mola
}  // namespace oneflow
