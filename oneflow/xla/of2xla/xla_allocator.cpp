#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "oneflow/xla/of2xla/xla_allocator.h"

namespace oneflow {
namespace mola {

size_t Align(int alignment, size_t size) {
  return (size + alignment - 1) / alignment * alignment;
}

XlaAllocator::XlaAllocator(const se::Platform* platform,
                           DeviceMemoryPool *memory_pool)
    : se::DeviceMemoryAllocator(platform), mem_pool_(memory_pool), offset_(0) {}

XlaAllocator::~XlaAllocator() {}

xla::StatusOr<se::OwningDeviceMemory> XlaAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  void* data = nullptr;
  if (size != 0) {
    CHECK_EQ(device_ordinal, mem_pool_->device_ordinal());
    data = mem_pool_->AllocateRaw(offset_, size);
    CHECK(data) << absl::StrCat("Out of memory while trying to allocate ",
                                size, " bytes.");
    offset_ += Align(32 /*alignment*/, size);
  }
  return se::OwningDeviceMemory(se::DeviceMemoryBase(data, size),
                                device_ordinal, this);
}

tensorflow::Status XlaAllocator::Deallocate(int device_ordinal,
                                            se::DeviceMemoryBase mem) {
  return tensorflow::Status::OK();
}

}  // namespace mola
}  // namespace oneflow
