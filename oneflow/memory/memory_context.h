#ifndef ONEFLOW_MEMORY_MEMORY_CONTEXT_H_
#define ONEFLOW_MEMORY_MEMORY_CONTEXT_H_

#include "common/id_map.h"

namespace oneflow {

enum class MemoryType {
  kHostPageableMemory = 0,
  kHostPinnedMemory,
  kDeviceMemory,
  kUnknown
};

struct MemoryContext {
  MemoryType mem_type;
  DeviceGlobalId device_global_id;
};

} // namespace oneflow

#endif // ONEFLOW_MEMORY_MEMORY_CONTEXT_H_
