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
#include "oneflow/cambricon/ep/mlu_device.h"

#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/common/mlu_guard.h"
#include "oneflow/cambricon/ep/mlu_event.h"
#include "oneflow/cambricon/ep/mlu_stream.h"

namespace oneflow {
namespace ep {

namespace {

constexpr size_t kDefaultConstBufElementCount = 1024 * 1024;

template<typename T>
void CreateConstBuffer(void** buf, T value, size_t n) {
  OF_MLU_CHECK(cnrtMalloc(buf, n * sizeof(T)));
  std::vector<T> host(n, value);
  OF_MLU_CHECK(cnrtMemcpy(*buf, host.data(), n * sizeof(T), cnrtMemcpyHostToDev));
}

}  // namespace

MluDevice::MluDevice(int device_index, DeviceManager* device_manager)
    : device_index_(device_index),
      event_flags_{},
      device_manager_(device_manager),
      const_buf_elem_cnt_(0),
      const_zeros_buffer_(nullptr),
      const_ones_buffer_fp32_(nullptr) {
  MluCurrentDeviceGuard guard(device_index_);
  event_flags_ = 0;
  const_buf_elem_cnt_ = ParseIntegerFromEnv("ONEFLOW_EP_MLU_CONST_BUFFER_ELEMENT_COUNT",
                                            kDefaultConstBufElementCount);
  if (const_buf_elem_cnt_ > 0) {
    CreateConstBuffer<float>(&const_zeros_buffer_, static_cast<float>(0), const_buf_elem_cnt_);
    CreateConstBuffer<float>(&const_ones_buffer_fp32_, static_cast<float>(1.0),
                             const_buf_elem_cnt_);
  }
  OF_MLU_CHECK(cnrtDeviceGetAttribute(&nclusters_, cnrtAttrClusterCount, device_index_));
  OF_MLU_CHECK(
      cnrtDeviceGetAttribute(&ncores_per_cluster_, cnrtAttrMcorePerCluster, device_index_));
}

MluDevice::~MluDevice() {
  MluCurrentDeviceGuard guard(device_index_);
  for (auto* event : events_) { delete event; }
  OF_MLU_CHECK(cnrtFree(const_zeros_buffer_));
  OF_MLU_CHECK(cnrtFree(const_ones_buffer_fp32_));
}

void MluDevice::SetAsActiveDevice() { OF_MLU_CHECK(cnrtSetDevice(device_index_)); }

void MluDevice::TryReset() {
  SetAsActiveDevice();
  OF_MLU_CHECK(cnrtDeviceReset());
}

Stream* MluDevice::CreateStream() {
  MluCurrentDeviceGuard guard(device_index_);
  return new MluStream(this);
}

void MluDevice::DestroyStream(Stream* stream) {
  MluCurrentDeviceGuard guard(device_index_);
  delete stream;
}

void MluDevice::CreateEvents(Event** events, size_t count) {
  size_t copied = 0;
  {
    std::lock_guard<std::mutex> lock(events_mutex_);
    copied = std::min(count, events_.size());
    size_t offset = events_.size() - copied;
    std::copy(events_.begin() + offset, events_.end(), events);
    events_.resize(offset);
  }
  if (copied != count) {
    MluCurrentDeviceGuard guard(device_index_);
    for (size_t i = copied; i < count; ++i) { events[i] = new MluEvent(event_flags_); }
  }
}

void MluDevice::DestroyEvents(Event** events, size_t count) {
  std::lock_guard<std::mutex> lock(events_mutex_);
  events_.insert(events_.end(), events, events + count);
}

Maybe<void> MluDevice::Alloc(const AllocationOptions& options, void** ptr, size_t size) {
  MluCurrentDeviceGuard guard(device_index_);
  CHECK(!options.HasPinnedDevice());
  cnrtRet_t err = cnrtMalloc(ptr, size);
  if (err != cnrtSuccess) {
    return Error::RuntimeError() << "MluDevice::Alloc error";
  } else {
    return Maybe<void>::Ok();
  }
}

void MluDevice::Free(const AllocationOptions& attr, void* ptr) {
  MluCurrentDeviceGuard guard(device_index_);
  OF_MLU_CHECK(cnrtFree(ptr));
}

Maybe<void> MluDevice::AllocPinned(const AllocationOptions& options, void** ptr, size_t size) {
  MluCurrentDeviceGuard guard(device_index_);
  cnrtRet_t err = cnrtHostMalloc(ptr, size);
  if (err != cnrtSuccess) {
    return Error::RuntimeError() << "MluDevice::AllocPinned error";
  } else {
    return Maybe<void>::Ok();
  }
}

void MluDevice::FreePinned(const AllocationOptions& options, void* ptr) {
  MluCurrentDeviceGuard guard(device_index_);
  OF_MLU_CHECK(cnrtFreeHost(ptr));
}

const void* MluDevice::GetConstZeros(DataType data_type, size_t n) const {
  if (GetSizeOfDataType(data_type) * n
      <= GetSizeOfDataType(DataType::kFloat) * const_buf_elem_cnt_) {
    return const_zeros_buffer_;
  } else {
    return nullptr;
  }
}

const void* MluDevice::GetConstOnes(DataType data_type, size_t n) const {
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
