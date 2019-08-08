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
    : se::DeviceMemoryAllocator(platform), allocator_(allocator), offset_(0) {}

XlaAllocator::~XlaAllocator() {}

xla::StatusOr<se::OwningDeviceMemory> XlaAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  void* data = nullptr;
  if (size != 0) {
    data = allocator_->AllocateRaw(offset_, size);
    CHECK(data) << absl::StrCat("Out of memory while trying to allocate ",
                                size, " bytes.");
    offset_ += Align(64/*alignment*/, size);
  }
  return se::OwningDeviceMemory(se::DeviceMemoryBase(data, size),
                                device_ordinal, this);
}

tensorflow::Status XlaAllocator::Deallocate(int device_ordinal,
                                            se::DeviceMemoryBase mem) {
  return tensorflow::Status::OK();
}

void XlaAllocator::ReserveWorkspace(size_t workspace_bytes) {
  allocator_->Reserve(workspace_bytes);
}

void XlaAllocator::LockWorkspace() {
  allocator_->Lock();
}

void XlaAllocator::UnlockWorkspace() {
  allocator_->Unlock();
}

}  // namespace mola
}  // namespace oneflow
