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

namespace oneflow {

namespace ep {

CudaDevice::CudaDevice(int device_index)
    : device_index_(device_index), event_flags_{}, properties_{} {
  CudaCurrentDeviceGuard guard(device_index_);
  OF_CUDA_CHECK(cudaGetDeviceProperties(&properties_, device_index_));
  event_flags_ = cudaEventDisableTiming;
  if (ParseBooleanFromEnv("ONEFLOW_STREAM_CUDA_EVENT_FLAG_BLOCKING_SYNC", false)) {
    event_flags_ |= cudaEventBlockingSync;
  }
}

CudaDevice::~CudaDevice() {
  CudaCurrentDeviceGuard guard(device_index_);
  for (auto* event : events_) { delete event; }
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

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA
