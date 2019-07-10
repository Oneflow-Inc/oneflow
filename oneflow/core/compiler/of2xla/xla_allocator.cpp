#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "oneflow/core/compiler/of2xla/xla_allocator.h"

namespace oneflow {
namespace mola {

XlaAllocator::XlaAllocator(const se::Platform* platform)
    : se::DeviceMemoryAllocator(platform) {}

XlaAllocator::~XlaAllocator() {}

xla::StatusOr<se::OwningDeviceMemory> XlaAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  tensorflow::AllocationAttributes attrs;
  attrs.no_retry_on_failure = !retry_on_failure;
  void* data = nullptr;
  if (size != 0) {
    // Empty object for lazily allocate memory
    data = tensorflow::XlaTensor::ToOpaquePointer(new tensorflow::XlaTensor());
    // CHECK(data) << absl::StrCat("Out of memory while trying to allocate ",
    //                            size, " bytes.");
  }
  return se::OwningDeviceMemory(se::DeviceMemoryBase(data, size),
                                device_ordinal, this);
}

tensorflow::Status XlaAllocator::Deallocate(int device_ordinal,
                                            se::DeviceMemoryBase mem) {
  delete tensorflow::XlaTensor::FromOpaquePointer(mem.opaque());
  return tensorflow::Status::OK();
}

}  // namespace mola
}  // namespace oneflow
