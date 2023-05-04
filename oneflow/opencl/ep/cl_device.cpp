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
#include "oneflow/opencl/ep/cl_device.h"

#include "oneflow/opencl/common/cl_util.h"
#include "oneflow/opencl/common/cl_guard.h"
#include "oneflow/opencl/ep/cl_event.h"
#include "oneflow/opencl/ep/cl_stream.h"

namespace oneflow {
namespace ep {

namespace {

constexpr size_t kDefaultConstBufElementCount = 1024 * 1024;

template<typename T>
void CreateConstBuffer(void** buf, T value, size_t n) {
  OF_CL_CHECK(clMalloc(buf, n * sizeof(T)));
  std::vector<T> host(n, value);
  OF_CL_CHECK(clMemcpy(*buf, host.data(), n * sizeof(T), MemcpyKind::kHtoD));
}

}  // namespace

clDevice::clDevice(int device_index, DeviceManager* device_manager)
    : device_index_(device_index),
      event_flags_{},
      device_manager_(device_manager),
      const_buf_elem_cnt_(0),
      const_zeros_buffer_(nullptr),
      const_ones_buffer_fp32_(nullptr) {
  clCurrentDeviceGuard guard(device_index_);
  event_flags_ = 0;
  const_buf_elem_cnt_ =
      ParseIntegerFromEnv("ONEFLOW_EP_CL_CONST_BUFFER_ELEMENT_COUNT", kDefaultConstBufElementCount);
  if (const_buf_elem_cnt_ > 0) {
    CreateConstBuffer<float>(&const_zeros_buffer_, static_cast<float>(0), const_buf_elem_cnt_);
    CreateConstBuffer<float>(&const_ones_buffer_fp32_, static_cast<float>(1.0),
                             const_buf_elem_cnt_);
  }
}

clDevice::~clDevice() {
  clCurrentDeviceGuard guard(device_index_);
  for (auto* event : events_) { delete event; }
  OF_CL_CHECK(clFree(const_zeros_buffer_));
  OF_CL_CHECK(clFree(const_ones_buffer_fp32_));
}

void clDevice::SetAsActiveDevice() { OF_CL_CHECK(clSetDevice(device_index_)); }

void clDevice::TryReset() {
  // TODO
}

Stream* clDevice::CreateStream() {
  clCurrentDeviceGuard guard(device_index_);
  return new clStream(this);
}

void clDevice::DestroyStream(Stream* stream) {
  clCurrentDeviceGuard guard(device_index_);
  if (stream) { delete stream; }
}

void clDevice::CreateEvents(Event** events, size_t count) {
  size_t copied = 0;
  {
    std::lock_guard<std::mutex> lock(events_mutex_);
    copied = std::min(count, events_.size());
    size_t offset = events_.size() - copied;
    std::copy(events_.begin() + offset, events_.end(), events);
    events_.resize(offset);
  }
  if (copied != count) {
    clCurrentDeviceGuard guard(device_index_);
    for (size_t i = copied; i < count; ++i) { events[i] = new clEvent(event_flags_); }
  }
}

void clDevice::DestroyEvents(Event** events, size_t count) {
  std::lock_guard<std::mutex> lock(events_mutex_);
  events_.insert(events_.end(), events, events + count);
}

Maybe<void> clDevice::Alloc(const AllocationOptions& options, void** ptr, size_t size) {
  clCurrentDeviceGuard guard(device_index_);
  CHECK(!options.HasPinnedDevice());
  cl_int err = clMalloc(ptr, size);
  if (err != CL_SUCCESS) {
    return Error::RuntimeError() << "clDevice::Alloc error";
  } else {
    return Maybe<void>::Ok();
  }
}

void clDevice::Free(const AllocationOptions& attr, void* ptr) {
  clCurrentDeviceGuard guard(device_index_);
  OF_CL_CHECK(clFree(ptr));
}

Maybe<void> clDevice::AllocPinned(const AllocationOptions& options, void** ptr, size_t size) {
  clCurrentDeviceGuard guard(device_index_);
  cl_int err = clMallocHost(ptr, size);
  if (err != CL_SUCCESS) {
    return Error::RuntimeError() << "clDevice::AllocPinned error";
  } else {
    return Maybe<void>::Ok();
  }
}

void clDevice::FreePinned(const AllocationOptions& options, void* ptr) {
  clCurrentDeviceGuard guard(device_index_);
  OF_CL_CHECK(clFreeHost(ptr));
}

const void* clDevice::GetConstZeros(DataType data_type, size_t n) const {
  if (GetSizeOfDataType(data_type) * n
      <= GetSizeOfDataType(DataType::kFloat) * const_buf_elem_cnt_) {
    return const_zeros_buffer_;
  } else {
    return nullptr;
  }
}

const void* clDevice::GetConstOnes(DataType data_type, size_t n) const {
  if (n <= const_buf_elem_cnt_) {
    if (data_type == DataType::kFloat) {
      return const_ones_buffer_fp32_;
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

}  // namespace ep
}  // namespace oneflow
