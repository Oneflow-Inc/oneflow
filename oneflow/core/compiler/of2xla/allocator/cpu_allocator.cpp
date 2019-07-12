#include "tensorflow/core/platform/mem.h"
#include "tensorflow/stream_executor/host/host_platform_id.h"
#include "oneflow/core/compiler/of2xla/xla_allocator.h"

namespace oneflow {
namespace mola {

// CPUAllocator directly borrows AlignedMalloc and AlignedFree of tensorflow
class CPUAllocator : public XlaAllocator {
 public:
  explicit CPUAllocator(const se::Platform* platform, int device_ordinal)
      : XlaAllocator(platform), device_ordinal_(device_ordinal) {}

  void* AllocateRaw(size_t alignment, size_t num_bytes) const override;

  void DeallocateRaw(se::DeviceMemoryBase mem) const override;

 private:
  int device_ordinal_;
};

REGISTER_XLA_ALLOCATOR(se::host::kHostPlatformId, CPUAllocator);

void *CPUAllocator::AllocateRaw(size_t alignment, size_t num_bytes) const {
  return tensorflow::port::AlignedMalloc(num_bytes, alignment);
}

void CPUAllocator::DeallocateRaw(se::DeviceMemoryBase mem) const {
  tensorflow::port::AlignedFree(mem.opaque());
}

}  // namespace mola
}  // namespace oneflow
