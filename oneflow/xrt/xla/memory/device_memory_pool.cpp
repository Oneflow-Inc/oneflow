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
#include "oneflow/xrt/xla/memory/device_memory_pool.h"
#include "oneflow/core/device/cuda_util.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/host/host_platform_id.h"

namespace oneflow {
namespace xrt {
namespace mola {

void DeviceMemoryPool::Reserve(size_t size) {
  if (limited_memory_size_ > -1) { CHECK_LT(size, limited_memory_size_); }

  while (size > capacity_) {
    Release();
    ReserveImpl(size);
  }
}

void DeviceMemoryPool::Release() {
  // Block host to ensure that all the launched kernels depend on this
  // memory buffer have been executed completely
  CHECK(stream_->BlockHostUntilDone().ok());

  ReleaseImpl();
}

auto DeviceMemoryPool::Registry()
    -> util::Registry<se::Platform::Id, std::function<DeviceMemoryPool*(se::Stream*, int)>>* {
  return util::Registry<se::Platform::Id,
                        std::function<DeviceMemoryPool*(se::Stream*, int)>>::Global();
}

std::shared_ptr<DeviceMemoryPool> DeviceMemoryPool::NewMemoryPool(const se::Platform* platform,
                                                                  se::Stream* stream,
                                                                  int device_ordinal) {
  return std::shared_ptr<DeviceMemoryPool>(
      DeviceMemoryPool::Registry()->Lookup(platform->id())(stream, device_ordinal));
}

namespace memory {

class CpuMemoryPool : public DeviceMemoryPool {
 public:
  explicit CpuMemoryPool(se::Stream* stream, int device_ordinal)
      : DeviceMemoryPool(stream, device_ordinal) {}
  virtual ~CpuMemoryPool() { Release(); }

 private:
  void ReserveImpl(size_t size) override {
    mem_buffer_ = new uint8_t[size];
    CHECK(mem_buffer_);
    capacity_ = size;
  }

  void ReleaseImpl() override {
    if (capacity_ > 0 && mem_buffer_) { delete[] mem_buffer_; }
    capacity_ = 0;
    mem_buffer_ = nullptr;
  }
};

REGISTER_XLA_MEMORY_POOL(se::host::kHostPlatformId, CpuMemoryPool);

class GpuMemoryPool : public DeviceMemoryPool {
 public:
  explicit GpuMemoryPool(se::Stream* stream, int device_ordinal)
      : DeviceMemoryPool(stream, device_ordinal) {}

  virtual ~GpuMemoryPool() { Release(); }

 private:
  void ReserveImpl(size_t size) override {
#ifdef WITH_CUDA
    int device_ordinal;
    cudaGetDevice(&device_ordinal);
    if (device_ordinal != device_ordinal_) { cudaSetDevice(device_ordinal_); }

    CudaCheck(cudaMalloc(&mem_buffer_, size));

    if (device_ordinal != device_ordinal_) { cudaSetDevice(device_ordinal); }
#else
    LOG(FATAL) << "Please recompile with CUDA.";
#endif
    CHECK(mem_buffer_);
    capacity_ = size;
  }

  void ReleaseImpl() override {
#ifdef WITH_CUDA
    int device_ordinal;
    cudaGetDevice(&device_ordinal);
    if (device_ordinal != device_ordinal_) { cudaSetDevice(device_ordinal_); }

    if (capacity_ > 0 && mem_buffer_) { CudaCheck(cudaFree(mem_buffer_)); }

    if (device_ordinal != device_ordinal_) { cudaSetDevice(device_ordinal); }
#else
    LOG(FATAL) << "Please recompile with CUDA.";
#endif
    capacity_ = 0;
    mem_buffer_ = nullptr;
  }
};

REGISTER_XLA_MEMORY_POOL(se::cuda::kCudaPlatformId, GpuMemoryPool);

}  // namespace memory

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
