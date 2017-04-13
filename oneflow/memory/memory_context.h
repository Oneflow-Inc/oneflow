#ifndef ONEFLOW_MEMORY_MEMORY_CONTEXT_H_
#define ONEFLOW_MEMORY_MEMORY_CONTEXT_H_

#include "job/id_manager.h"

namespace oneflow {

enum class MemoryType {
  kHostPageableMemory = 0,
  kHostPinnedMemory,
  kDeviceMemory,
  kUnknown
};

struct MemoryContext {
  MemoryType mem_type;
  MachineId machine_id;
  DevicePhysicalId device_physical_id;
};

} // namespace oneflow

#endif // ONEFLOW_MEMORY_MEMORY_CONTEXT_H_
