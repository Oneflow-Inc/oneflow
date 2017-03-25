#ifndef _MEMORY_MEMORY_MANAGER_H_
#define _MEMORY_MEMORY_MANAGER_H_

#include <cstdint>
#include <vector>
#include <array>

namespace oneflow {

enum class DeviceType {
  kCPU = 0,
  kCPUPinned,
  kGPU,
  kUnknown
};

enum class MemoryType {
  kHostPageableMemory = 0,
  kHostPinnedMemory,
  kDeviceMemory,
  kUnknown
};

// Memory Manager is used for global memory allocation
class MemoryManager {
 public:
  struct Context {
    DeviceType dev_type;
    int32_t dev_id;

    Context() : dev_type(DeviceType::kCPU), dev_id(0) {}
    Context(DeviceType dev_type, int32_t dev_id) :
      dev_type(dev_type), dev_id(dev_id) {}
  };

  struct Handle {
    void* dptr;
    size_t size;
    Context ctx;
  };

  struct DeviceMemoryInfo {
    size_t alloc_mem_size;
    size_t free_mem_size;
    size_t phy_mem_size;
  };

  virtual Handle Alloc(size_t size, Context ctx) = 0;

  virtual void Free(Handle handle) = 0;

  ~MemoryManager() = default;

  static MemoryManager* Get();
};
}  // namespace oneflow
#endif  // _MEMORY_MEMORY_MANAGER_H_
