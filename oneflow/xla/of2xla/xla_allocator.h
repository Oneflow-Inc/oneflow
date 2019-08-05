#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ALLOCATOR_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ALLOCATOR_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "oneflow/xla/of2xla/memory/memory_pool.h"

namespace oneflow {
namespace mola {

namespace se = tensorflow::se;
using uint64 = tensorflow::uint64;

class XlaAllocator : public se::DeviceMemoryAllocator {
 public:  
  explicit XlaAllocator(const se::Platform* platform,
                        DeviceMemoryPool *memory_pool);
  virtual ~XlaAllocator();

  DeviceMemoryPool *memory_pool() const { return mem_pool_; }

  xla::StatusOr<se::OwningDeviceMemory> Allocate(
      int device_ordinal, uint64 size, bool retry_on_failure) override;
  tensorflow::Status Deallocate(int device_ordinal,
                                se::DeviceMemoryBase mem) override;

  bool AllowsAsynchronousDeallocation() const override { return true; }

 private:
  DeviceMemoryPool *mem_pool_;
  size_t offset_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ALLOCATOR_H_
