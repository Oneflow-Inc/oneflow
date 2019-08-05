#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_MEMORY_MEMORY_POOL_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_MEMORY_MEMORY_POOL_H_

#include "glog/logging.h"

namespace oneflow {
namespace mola {

class DeviceMemoryPool {
 public:
  explicit DeviceMemoryPool(int device_ordinal)
      : mem_buffer_(nullptr), capacity_(0), device_ordinal_(device_ordinal) {}

  virtual ~DeviceMemoryPool() {}

  int device_ordinal() const { return device_ordinal_; }
  size_t capacity() const { return capacity_; }

  virtual void Reserve(size_t size) = 0;

  virtual void Release() = 0;

  virtual void *AllocateRaw(size_t offset, size_t size) {
    CHECK_LT(offset + size, capacity_);
    return reinterpret_cast<void *>(mem_buffer_ + offset);
  }

  static DeviceMemoryPool *NewCpuMemoryPool(int device_ordinal);
  static DeviceMemoryPool *NewGpuMemoryPool(const void *cuda_stream,
                                            int device_ordinal);
 protected:
  uint8_t *mem_buffer_;

  size_t capacity_;

  int device_ordinal_;

  // Limited size for buffer size. If limit is not requared, then set it -1
  int64_t limited_memory_size_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_MEMORY_MEMORY_POOL_H_
