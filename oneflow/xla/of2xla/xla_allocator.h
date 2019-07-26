#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ALLOCATOR_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ALLOCATOR_H_

#include <mutex>
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/core/framework/allocator.h"

namespace std {
template<>
struct hash<std::pair<void *, int>> {
  size_t operator()(const std::pair<void *, int> &val) const {
    uint64_t first = reinterpret_cast<uint64_t>(val.first);
    return std::hash<uint64_t>()(first) ^ std::hash<int>()(val.second);
//    return std::hash<void *>(val.first) ^ std::hash<int>()(val.second);
  }
};
}  // namespace std

namespace oneflow {
namespace mola {

namespace se = tensorflow::se;
using uint64 = tensorflow::uint64;

class XlaAllocator : public se::DeviceMemoryAllocator {
 public:  
  explicit XlaAllocator(const se::Platform* platform);
  virtual ~XlaAllocator();

  xla::StatusOr<se::OwningDeviceMemory> Allocate(
      int device_ordinal, uint64 size, bool retry_on_failure) override;
  tensorflow::Status Deallocate(int device_ordinal,
                                se::DeviceMemoryBase mem) override;

  bool AllowsAsynchronousDeallocation() const override { return true; }

  virtual void* AllocateRaw(size_t alignment, size_t num_bytes) const;
  virtual void DeallocateRaw(se::DeviceMemoryBase mem) const;
 
  typedef std::function<XlaAllocator *(const se::Platform *,
                                       int device_ordinal)> AllocatorFactory;

 private:
  XlaAllocator *LookupAllocator(int device_ordinal);

  static XlaAllocator *CreateAllocator(const se::Platform *platform,
                                       int device_ordinal);
  static void Register(const se::Platform::Id &platform_id,
                       AllocatorFactory factory);

  template <typename T>
  friend class XlaAllocatorRegistarr;

  mutable std::mutex mutex_;

  const se::Platform* platform_;
  std::unordered_map<std::pair<se::Platform::Id, int>, XlaAllocator *>
      allocators_;
};

template <typename AllocatorClass>
class XlaAllocatorRegistarr {
 public:
  XlaAllocatorRegistarr(const se::Platform::Id &platform_id) {
    auto factory = [](const se::Platform* platform, int device_ordinal)
         -> AllocatorClass* {
      return new AllocatorClass(platform, device_ordinal);
    };
    XlaAllocator::Register(platform_id, factory);
  }
};

#define REGISTER_XLA_ALLOCATOR(PlatformId, AllocatorClass)         \
  static XlaAllocatorRegistarr<AllocatorClass>                     \
      _xla_allocator_##AllocatorClass##_ __attribute__((unused)) = \
      XlaAllocatorRegistarr<AllocatorClass>(PlatformId)

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ALLOCATOR_H_
