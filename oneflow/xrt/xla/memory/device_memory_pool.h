/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_XRT_XLA_MEMORY_DEVICE_MEMORY_POOL_H_
#define ONEFLOW_XRT_XLA_MEMORY_DEVICE_MEMORY_POOL_H_

#include <functional>
#include "glog/logging.h"

#include "oneflow/xrt/utility/registry.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream.h"

namespace oneflow {
namespace xrt {
namespace mola {

namespace se = tensorflow::se;

class DeviceMemoryPool {
 public:
  DeviceMemoryPool() = delete;
  virtual ~DeviceMemoryPool() = default;

  virtual void* AllocateRaw(size_t offset, size_t size) {
    CHECK_LE(offset + size, capacity_);
    return reinterpret_cast<void*>(mem_buffer_ + offset);
  }

  void Reserve(size_t size);
  void Release();

  size_t capacity() const { return capacity_; }

  int device_ordinal() const { return device_ordinal_; }

  static std::shared_ptr<DeviceMemoryPool> NewMemoryPool(const se::Platform* platform,
                                                         se::Stream* stream, int device_ordinal);

  static auto Registry()
      -> util::Registry<se::Platform::Id, std::function<DeviceMemoryPool*(se::Stream*, int)>>*;

 protected:
  explicit DeviceMemoryPool(se::Stream* stream, int device_ordinal)
      : mem_buffer_(nullptr), capacity_(0), stream_(stream), device_ordinal_(device_ordinal) {}

  virtual void ReserveImpl(size_t size) = 0;
  virtual void ReleaseImpl() = 0;

 protected:
  uint8_t* mem_buffer_ = nullptr;
  size_t capacity_ = 0;

  se::Stream* stream_ = nullptr;
  int device_ordinal_ = 0;
  // Limited size for allocated buffer. Set -1 if limit is not required.
  int64_t limited_memory_size_ = -1;
};

template<typename MemoryPool>
class DeviceMemoryPoolRegistarr {
 public:
  DeviceMemoryPoolRegistarr(const se::Platform::Id& platform_id) {
    DeviceMemoryPool::Registry()->Register(platform_id, [](se::Stream* stream, int device_ordinal) {
      return new MemoryPool(stream, device_ordinal);
    });
  }
};

#define REGISTER_XLA_MEMORY_POOL(PlatformId, MemoryPool)                           \
  static DeviceMemoryPoolRegistarr<MemoryPool> _device_memory_pool_##MemoryPool##_ \
      __attribute__((unused)) = DeviceMemoryPoolRegistarr<MemoryPool>(PlatformId)

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_MEMORY_DEVICE_MEMORY_POOL_H_
