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
#ifdef WITH_NPU
#include "oneflow/core/ep/npu/npu_device.h"
#include "oneflow/core/ep/npu/npu_event.h"
#include "oneflow/core/ep/npu/npu_stream.h"

namespace oneflow {

namespace ep {

namespace {

constexpr size_t kDefaultConstBufElementCount = 1024 * 1024;

template<typename T>
void CreateConstBuffer(void** buf, T value, size_t n) {
  std::cout<<"CreateConstBuffer"<<std::endl;
  OF_NPU_CHECK(aclrtMalloc(buf, n * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<T> host(n, value);
  OF_NPU_CHECK(aclrtMemcpy(*buf, n * sizeof(T), host.data(), n * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));
}

}  // namespace

NpuDevice::NpuDevice(int device_index, DeviceManager* device_manager)
    : device_index_(device_index),
      event_flags_{},
      device_manager_(device_manager),
      const_buf_elem_cnt_(0),
      const_zeros_buffer_(nullptr),
      const_ones_buffer_fp32_(nullptr),
      const_ones_buffer_fp16_(nullptr) {
  std::cout<<"NpuDevice::NpuDevice(int,DM)"<<std::endl;
  NpuCurrentDeviceGuard guard(device_index_);
  const_buf_elem_cnt_ = ParseIntegerFromEnv("ONEFLOW_EP_NPU_CONST_BUFFER_ELEMENT_COUNT",
                                            kDefaultConstBufElementCount);
  if (const_buf_elem_cnt_ > 0) {
    CreateConstBuffer<float>(&const_zeros_buffer_, static_cast<float>(0), const_buf_elem_cnt_);
    CreateConstBuffer<float>(&const_ones_buffer_fp32_, static_cast<float>(1.0),
                             const_buf_elem_cnt_);
    CreateConstBuffer<aclFloat16>(&const_ones_buffer_fp16_, static_cast<aclFloat16>(1.0), const_buf_elem_cnt_);
  }
}

NpuDevice::~NpuDevice() {
  std::cout<<"NpuDevice::~NpuDevice()"<<std::endl;
  NpuCurrentDeviceGuard guard(device_index_);
  //for (auto* event : events_) { delete event; }
  OF_NPU_CHECK(aclrtFree(const_zeros_buffer_));
  OF_NPU_CHECK(aclrtFree(const_ones_buffer_fp32_));
  OF_NPU_CHECK(aclrtFree(const_ones_buffer_fp16_));
}

void NpuDevice::SetAsActiveDevice() { }

Stream* NpuDevice::CreateStream() {
  std::cout<<"NpuDevice::CreateStream()"<<std::endl;
  NpuCurrentDeviceGuard guard(device_index_);
  return new NpuStream(this);
}

void NpuDevice::DestroyStream(Stream* stream) {
  std::cout<<"NpuDevice::DestroyStream()"<<std::endl;
  NpuCurrentDeviceGuard guard(device_index_);
  delete stream;
}

void NpuDevice::CreateEvents(Event** events, size_t count) {
  std::cout<<"NpuDevice::CreateEvents()"<<std::endl;
  size_t copied = 0;
  {
    std::lock_guard<std::mutex> lock(events_mutex_);
    copied = std::min(count, events_.size());
    size_t offset = events_.size() - copied;
    std::copy(events_.begin() + offset, events_.end(), events);
    events_.resize(offset);
  }
  if (copied != count) {
    NpuCurrentDeviceGuard guard(device_index_);
    for (size_t i = copied; i < count; ++i) { events[i] = new NpuEvent(event_flags_); }
  }
}

void NpuDevice::DestroyEvents(Event** events, size_t count) {
  std::cout<<"NpuDevice::DestroyEvents()"<<std::endl;
  std::lock_guard<std::mutex> lock(events_mutex_);
  events_.insert(events_.end(), events, events + count);
}

Maybe<void> NpuDevice::Alloc(const AllocationOptions& options, void** ptr, size_t size) {
  std::cout<<"NpuDevice::Alloc()"<<std::endl;
  NpuCurrentDeviceGuard guard(device_index_);
  CHECK(!options.HasPinnedDevice());
  aclError err = aclrtMalloc(ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (err != ACL_SUCCESS) {
    return Error::RuntimeError() << err;
  } else {
    return Maybe<void>::Ok();
  }
}

void NpuDevice::Free(const AllocationOptions& attr, void* ptr) {
  std::cout<<"NpuDevice::Free()"<<std::endl;
  NpuCurrentDeviceGuard guard(device_index_);
  OF_NPU_CHECK(aclrtFree(ptr));
}

const void* NpuDevice::GetConstZeros(DataType data_type, size_t n) const {
  std::cout<<"NpuDevice::GetConstZeros()"<<std::endl;
  if (GetSizeOfDataType(data_type) * n
      <= GetSizeOfDataType(DataType::kFloat) * const_buf_elem_cnt_) {
    return const_zeros_buffer_;
  } else {
    return nullptr;
  }
}

const void* NpuDevice::GetConstOnes(DataType data_type, size_t n) const {
  std::cout<<"NpuDevice::GetConstOnes()"<<std::endl;
  if (n <= const_buf_elem_cnt_) {
    if (data_type == DataType::kFloat) {
      return const_ones_buffer_fp32_;
    } else if (data_type == DataType::kFloat16) {
      return const_ones_buffer_fp16_;
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_NPU
