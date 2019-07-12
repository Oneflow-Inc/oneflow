#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "oneflow/core/compiler/of2xla/xla_allocator.h"

namespace oneflow {
namespace mola {

typedef XlaAllocator::AllocatorFactory AllocatorFactory;
typedef std::unordered_map<se::Platform::Id, AllocatorFactory>
    AllocatorFactoryMap;

static AllocatorFactoryMap* GlobalAllocatorFactories() {
  static AllocatorFactoryMap factories;
  return &factories;
}

void XlaAllocator::Register(const se::Platform::Id &platform_id,
                            AllocatorFactory factory) {
  AllocatorFactoryMap *factories = GlobalAllocatorFactories();
  if (factories->count(platform_id)) {
    DLOG(WARNING) << "Allocator factory for platform id (" << platform_id
                  << ") has been registed more than once";
  }
  factories->emplace(platform_id, factory);
}

XlaAllocator *XlaAllocator::CreateAllocator(
    const se::Platform *platform, int device_ordinal) {
  AllocatorFactoryMap *factories = GlobalAllocatorFactories();
  auto it = factories->find(platform->id());
  if (it == factories->end()) {
    LOG(FATAL) << "Allocator factory for platform id (" << platform->id()
               << ") has not been registed.";
    return nullptr;
  }
  return (it->second)(platform, device_ordinal);
}

XlaAllocator::XlaAllocator(const se::Platform* platform)
    : se::DeviceMemoryAllocator(platform), platform_(platform) {}

XlaAllocator::~XlaAllocator() {}

xla::StatusOr<se::OwningDeviceMemory> XlaAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  XlaAllocator *allocator = LookupAllocator(device_ordinal);
  void* data = nullptr;
  if (size != 0) {
    data = allocator->AllocateRaw(32 /*alignment*/, size);
    CHECK(data) << absl::StrCat("Out of memory while trying to allocate ",
                                size, " bytes.");
  }
  return se::OwningDeviceMemory(se::DeviceMemoryBase(data, size),
                                device_ordinal, this);
}

tensorflow::Status XlaAllocator::Deallocate(int device_ordinal,
                                            se::DeviceMemoryBase mem) {
  XlaAllocator *allocator = LookupAllocator(device_ordinal);
  allocator->DeallocateRaw(mem);
  return tensorflow::Status::OK();
}

void* XlaAllocator::AllocateRaw(size_t alignment, size_t num_bytes) const {
  LOG(FATAL) << "Should not call base class AllocateRaw.";
  return nullptr;
}

void XlaAllocator::DeallocateRaw(se::DeviceMemoryBase mem) const {
  LOG(FATAL) << "Should not call base class DeallocateRaw.";
}

XlaAllocator *XlaAllocator::LookupAllocator(int device_ordinal) {
  auto allocator_key = std::make_pair(platform_->id(), device_ordinal);
  XlaAllocator *allocator = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocators_.find(allocator_key);
    if (it == allocators_.end()) {
      it = allocators_.emplace(
          allocator_key,
          XlaAllocator::CreateAllocator(platform_, device_ordinal)).first;
    }
    allocator = it->second;
  }
  return allocator;
}

}  // namespace mola
}  // namespace oneflow
