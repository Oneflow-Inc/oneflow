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
#include "oneflow/core/ep/cuda/cuda_device.h"
#include "oneflow/core/ep/cuda/cuda_event.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

#ifdef WITH_CUDA

#include <cuda.h>
#include <cuda_fp16.h>

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif

namespace oneflow {

namespace ep {

namespace {

constexpr size_t kDefaultConstBufElementCount = 1024 * 1024;

template<typename T>
void CreateConstBuffer(void** buf, T value, size_t n) {
  OF_CUDA_CHECK(cudaMalloc(buf, n * sizeof(T)));
  std::vector<T> host(n, value);
  OF_CUDA_CHECK(cudaMemcpy(*buf, host.data(), n * sizeof(T), cudaMemcpyDefault));
}

}  // namespace

CudaDevice::CudaDevice(int device_index, DeviceManager* device_manager)
    : device_index_(device_index),
      event_flags_{},
      properties_{},
      device_manager_(device_manager),
      const_buf_elem_cnt_(0),
      const_zeros_buffer_(nullptr),
      const_ones_buffer_fp32_(nullptr),
      const_ones_buffer_fp16_(nullptr),
      const_ones_buffer_bf16_(nullptr) {
  CudaCurrentDeviceGuard guard(device_index_);
  OF_CUDA_CHECK(cudaGetDeviceProperties(&properties_, device_index_));
  {
    const char* env_name = "ONEFLOW_EP_CUDA_DEVICE_FLAGS";
    if (std::getenv(env_name) != nullptr) {
      const unsigned int flags = ParseIntegerFromEnv(env_name, 0);
      OF_CUDA_CHECK(cudaSetDeviceFlags(flags));
    }
  }
  event_flags_ = cudaEventDisableTiming;
  if (ParseBooleanFromEnv("ONEFLOW_STREAM_CUDA_EVENT_FLAG_BLOCKING_SYNC", false)) {
    event_flags_ |= cudaEventBlockingSync;
  }
  const_buf_elem_cnt_ = ParseIntegerFromEnv("ONEFLOW_EP_CUDA_CONST_BUFFER_ELEMENT_COUNT",
                                            kDefaultConstBufElementCount);
  if (const_buf_elem_cnt_ > 0) {
    CreateConstBuffer<float>(&const_zeros_buffer_, static_cast<float>(0), const_buf_elem_cnt_);
    CreateConstBuffer<float>(&const_ones_buffer_fp32_, static_cast<float>(1.0),
                             const_buf_elem_cnt_);
    CreateConstBuffer<half>(&const_ones_buffer_fp16_, static_cast<half>(1.0), const_buf_elem_cnt_);
#if CUDA_VERSION >= 11000
    CreateConstBuffer<nv_bfloat16>(&const_ones_buffer_bf16_, static_cast<nv_bfloat16>(1.0),
                                   const_buf_elem_cnt_);
#endif
  }
}

CudaDevice::~CudaDevice() {
  CudaCurrentDeviceGuard guard(device_index_);
  for (auto* event : events_) { delete event; }
  OF_CUDA_CHECK(cudaFree(const_zeros_buffer_));
  OF_CUDA_CHECK(cudaFree(const_ones_buffer_fp32_));
  OF_CUDA_CHECK(cudaFree(const_ones_buffer_fp16_));
  OF_CUDA_CHECK(cudaFree(const_ones_buffer_bf16_));
}

void CudaDevice::SetAsActiveDevice() { OF_CUDA_CHECK(cudaSetDevice(device_index_)); }

Stream* CudaDevice::CreateStream() {
  CudaCurrentDeviceGuard guard(device_index_);
  return new CudaStream(this);
}

void CudaDevice::DestroyStream(Stream* stream) {
  CudaCurrentDeviceGuard guard(device_index_);
  delete stream;
}

void CudaDevice::CreateEvents(Event** events, size_t count) {
  size_t copied = 0;
  {
    std::lock_guard<std::mutex> lock(events_mutex_);
    copied = std::min(count, events_.size());
    size_t offset = events_.size() - copied;
    std::copy(events_.begin() + offset, events_.end(), events);
    events_.resize(offset);
  }
  if (copied != count) {
    CudaCurrentDeviceGuard guard(device_index_);
    for (size_t i = copied; i < count; ++i) { events[i] = new CudaEvent(event_flags_); }
  }
}

void CudaDevice::DestroyEvents(Event** events, size_t count) {
  std::lock_guard<std::mutex> lock(events_mutex_);
  events_.insert(events_.end(), events, events + count);
}

Maybe<void> CudaDevice::Alloc(const AllocationOptions& options, void** ptr, size_t size) {
  CudaCurrentDeviceGuard guard(device_index_);
  CHECK(!options.HasPinnedDevice());
  cudaError_t err = cudaMalloc(ptr, size);
  if (err != cudaSuccess) {
    if (err == cudaErrorMemoryAllocation) {
      // NOTE:return out of memory error, so vm will try to shrink memory and rerun
      return Error::OutOfMemoryError() << cudaGetErrorString(err);
    }
    return Error::RuntimeError() << cudaGetErrorString(err);
  } else {
    return Maybe<void>::Ok();
  }
}

void CudaDevice::Free(const AllocationOptions& attr, void* ptr) {
  CudaCurrentDeviceGuard guard(device_index_);
  OF_CUDA_CHECK(cudaFree(ptr));
}

Maybe<void> CudaDevice::AllocPinned(const AllocationOptions& options, void** ptr, size_t size) {
  CudaCurrentDeviceGuard guard(device_index_);
  cudaError_t err = NumaAwareCudaMallocHost(device_index_, ptr, size);
  if (err != cudaSuccess) {
    return Error::RuntimeError() << cudaGetErrorString(err);
  } else {
    return Maybe<void>::Ok();
  }
}

void CudaDevice::FreePinned(const AllocationOptions& options, void* ptr) {
  CudaCurrentDeviceGuard guard(device_index_);
  OF_CUDA_CHECK(cudaFreeHost(ptr));
}

const cudaDeviceProp& CudaDevice::properties() const { return properties_; }

const void* CudaDevice::GetConstZeros(DataType data_type, size_t n) const {
  if (GetSizeOfDataType(data_type) * n
      <= GetSizeOfDataType(DataType::kFloat) * const_buf_elem_cnt_) {
    return const_zeros_buffer_;
  } else {
    return nullptr;
  }
}

const void* CudaDevice::GetConstOnes(DataType data_type, size_t n) const {
  if (n <= const_buf_elem_cnt_) {
    if (data_type == DataType::kFloat) {
      return const_ones_buffer_fp32_;
    } else if (data_type == DataType::kFloat16) {
      return const_ones_buffer_fp16_;
    } else if (data_type == DataType::kBFloat16) {
      return const_ones_buffer_bf16_;
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA
