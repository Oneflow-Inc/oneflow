#ifndef ONEFLOW_ENGINE_XLA_OF2XLA_MEMORY_DEVICE_BUFFER_ALLOCATOR_H_
#define ONEFLOW_ENGINE_XLA_OF2XLA_MEMORY_DEVICE_BUFFER_ALLOCATOR_H_

#include <mutex>
#include <condition_variable>

#include "oneflow/engine/xla/of2xla/memory/device_memory_pool.h"

namespace oneflow {
namespace mola {

class DeviceBufferAllocator {
 public:
  explicit DeviceBufferAllocator(std::shared_ptr<DeviceMemoryPool> mem_pool)
      : mem_pool_(mem_pool) {
    // mem_pool_->Reserve(256 * 1024 * 1024/*256MiB*/);
  }

  virtual ~DeviceBufferAllocator() {}

  void *AllocateRaw(size_t offset, size_t size) {
    return mem_pool_->AllocateRaw(offset, size);
  }

  void Reserve(size_t size) {
    while (size > mem_pool_->capacity()) {
      std::unique_lock<std::mutex> lock(mutex_);
      cond_.wait(lock, [&]() { return lock_count_ == 0; });

      mem_pool_->Reserve(size);
    }
  }

  void Lock() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++lock_count_;
  }

  void Unlock() {
    std::unique_lock<std::mutex> lock(mutex_);
    --lock_count_;
    cond_.notify_all();
  }

 private:
  volatile uint64_t lock_count_ = 0;

  std::mutex mutex_;
 
  std::condition_variable cond_;

  std::shared_ptr<DeviceMemoryPool> mem_pool_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_XLA_OF2XLA_MEMORY_DEVICE_BUFFER_ALLOCATOR_H_
