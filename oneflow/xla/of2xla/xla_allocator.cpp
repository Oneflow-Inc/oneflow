#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "oneflow/xla/of2xla/xla_allocator.h"

namespace oneflow {
namespace mola {

inline size_t Align(int alignment, size_t size) {
  return (size + alignment - 1) / alignment * alignment;
}

XlaAllocator::XlaAllocator(const se::Platform* platform,
                           DeviceBufferAllocator *allocator)
    : se::DeviceMemoryAllocator(platform), allocator_(allocator),
      allocate_offset_(0), allocate_index_(0) {}

XlaAllocator::~XlaAllocator() {}

xla::StatusOr<se::OwningDeviceMemory> XlaAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  se::DeviceMemoryBase memory_base;
  if (allocate_index_ < populated_allocation_.size() &&
      populated_allocation_[allocate_index_].populated) {
    int index = populated_allocation_[allocate_index_].index;
    DCHECK_LT(index, populated_buffers_.size());
    memory_base = populated_buffers_[index];
  } else {
    void* data = nullptr;
    if (size != 0) {
      data = allocator_->AllocateRaw(allocate_offset_, size);
      allocate_offset_ += Align(64/*alignment*/, size);
    }
    memory_base = se::DeviceMemoryBase(data, size);
  }
  CHECK_EQ(memory_base.size(), size);
  allocate_index_++;
  return se::OwningDeviceMemory(memory_base, device_ordinal, this);
}

tensorflow::Status XlaAllocator::Deallocate(int device_ordinal,
                                            se::DeviceMemoryBase mem) {
  return tensorflow::Status::OK();
}

void XlaAllocator::ResetState() {
  allocate_offset_ = 0;
  allocate_index_ = 0;
}

void XlaAllocator::LockWorkspace() {
  allocator_->Lock();
}

void XlaAllocator::UnlockWorkspace() {
  allocator_->Unlock();
}

void XlaAllocator::ReserveWorkspace(size_t workspace_bytes) {
  allocator_->Reserve(workspace_bytes);
}

void XlaAllocator::PopulateDeviceMemory(
      const std::vector<se::DeviceMemoryBase> &device_buffers,
      const std::vector<int64_t> &allocation_indices) {
  populated_buffers_ = device_buffers;
  int64_t max_populated_index = 0;
  for (int i = 0; i < allocation_indices.size(); ++i) {
    int64_t index = allocation_indices[i]; 
    max_populated_index = std::max(max_populated_index, index);
  }

  populated_allocation_.resize(max_populated_index + 1);
  for (int i = 0; i < allocation_indices.size(); ++i) {
    int64_t index = allocation_indices[i];
    if (index >= 0) {
      populated_allocation_[index].populated = true;
      populated_allocation_[index].index = i;
    }
  }
}

}  // namespace mola
}  // namespace oneflow
