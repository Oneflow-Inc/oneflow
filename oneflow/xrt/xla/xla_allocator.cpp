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
#include "absl/strings/str_cat.h"
#include "glog/logging.h"

#include "oneflow/xrt/xla/xla_allocator.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace oneflow {
namespace xrt {
namespace mola {

inline size_t Align(int alignment, size_t size) {
  return (size + alignment - 1) / alignment * alignment;
}

XlaAllocator::XlaAllocator(const se::Platform* platform, DeviceBufferAllocator* allocator)
    : se::DeviceMemoryAllocator(platform),
      allocator_(allocator),
      allocate_offset_(0),
      allocate_index_(0) {}

XlaAllocator::~XlaAllocator() {}

xla::StatusOr<se::OwningDeviceMemory> XlaAllocator::Allocate(int device_ordinal, uint64 size,
                                                             bool retry_on_failure,
                                                             int64 /*memory_space*/) {
  se::DeviceMemoryBase memory_base;
  if (allocate_index_ < populated_buffers_.size()
      && populated_buffers_[allocate_index_].populated) {
    memory_base = populated_buffers_[allocate_index_].memory;
  } else {
    void* data = nullptr;
    if (size != 0) {
      data = allocator_->AllocateRaw(allocate_offset_, size);
      allocate_offset_ += Align(64 /*alignment*/, size);
    }
    memory_base = se::DeviceMemoryBase(data, size);
  }
  CHECK_EQ(memory_base.size(), size);
  allocate_index_++;
  return se::OwningDeviceMemory(memory_base, device_ordinal, this);
}

tensorflow::Status XlaAllocator::Deallocate(int device_ordinal, se::DeviceMemoryBase mem) {
  return tensorflow::Status::OK();
}

void XlaAllocator::ResetState() {
  allocate_offset_ = 0;
  allocate_index_ = 0;
}

void XlaAllocator::LockWorkspace() { allocator_->Lock(); }

void XlaAllocator::UnlockWorkspace() { allocator_->Unlock(); }

void XlaAllocator::ReserveWorkspace(size_t workspace_bytes) {
  allocator_->Reserve(workspace_bytes);
}

void XlaAllocator::PopulateDeviceMemory(const std::vector<se::DeviceMemoryBase>& device_buffers,
                                        const std::vector<int64_t>& allocation_indices) {
  int64_t max_populated_index = 0;
  for (int i = 0; i < allocation_indices.size(); ++i) {
    int64_t index = allocation_indices[i];
    max_populated_index = std::max(max_populated_index, index);
  }

  populated_buffers_.resize(max_populated_index + 1);
  for (int i = 0; i < allocation_indices.size(); ++i) {
    int64_t index = allocation_indices[i];
    if (index >= 0) {
      populated_buffers_[index].populated = true;
      populated_buffers_[index].index = i;
      populated_buffers_[index].memory = device_buffers[i];
    }
  }
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
