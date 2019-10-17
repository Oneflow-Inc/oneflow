#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ALLOCATOR_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ALLOCATOR_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

#include "oneflow/xrt/of2xla/memory/device_buffer_allocator.h"

namespace oneflow {
namespace mola {

namespace se = tensorflow::se;
using uint64 = tensorflow::uint64;

class XlaAllocator : public se::DeviceMemoryAllocator {
 public:
  explicit XlaAllocator(const se::Platform *platform,
                        DeviceBufferAllocator *allocator);
  virtual ~XlaAllocator();

  xla::StatusOr<se::OwningDeviceMemory> Allocate(
      int device_ordinal, uint64 size, bool retry_on_failure) override;
  tensorflow::Status Deallocate(int device_ordinal,
                                se::DeviceMemoryBase mem) override;

  bool AllowsAsynchronousDeallocation() const override { return true; }

  void ResetState();
  void ReserveWorkspace(size_t workspace_bytes);
  void LockWorkspace();
  void UnlockWorkspace();

  void PopulateDeviceMemory(
      const std::vector<se::DeviceMemoryBase> &device_buffers,
      const std::vector<int64_t> &allocation_indices);

 private:
  DeviceBufferAllocator *allocator_;
  size_t allocate_offset_;
  int64_t allocate_index_;

  struct AllocationBuffer {
    bool populated = false;
    int64_t index = -1;
    se::DeviceMemoryBase memory;
  };
  std::vector<AllocationBuffer> populated_buffers_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ALLOCATOR_H_
