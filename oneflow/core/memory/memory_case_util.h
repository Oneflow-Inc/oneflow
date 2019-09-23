#ifndef ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_

#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

inline bool operator==(const MemoryCase& lhs, const MemoryCase& rhs) {
  if (lhs.has_host_mem() && rhs.has_host_mem()) {
    const HostMemory& lhs_host_mem = lhs.host_mem();
    const HostMemory& rhs_host_mem = rhs.host_mem();
    if (lhs_host_mem.has_cuda_pinned_mem() && rhs_host_mem.has_cuda_pinned_mem()) {
      return lhs_host_mem.cuda_pinned_mem().device_id()
             == rhs_host_mem.cuda_pinned_mem().device_id();
    } else {
      return (!lhs_host_mem.has_cuda_pinned_mem()) && (!rhs_host_mem.has_cuda_pinned_mem());
    }
  }
  if (lhs.has_device_cuda_mem() && rhs.has_device_cuda_mem()) {
    return lhs.device_cuda_mem().device_id() == rhs.device_cuda_mem().device_id();
  }
  return false;
}

struct MemoryCaseUtil {
  static bool GetCommonMemoryCase(const MemoryCase& a, const MemoryCase& b, MemoryCase* common);
  static MemoryCase GetHostPinnedMemoryCaseForRegstSeparatedHeader(const MemoryCase& mem_case);
};

}  // namespace oneflow

namespace std {
template<>
struct hash<oneflow::MemoryCase> {
  size_t operator()(const oneflow::MemoryCase& val) const {
    if (val.has_host_mem()) {
      if (val.host_mem().has_cuda_pinned_mem()) {
        return 1025 + val.host_mem().cuda_pinned_mem().device_id();
      } else {
        return 1024;
      }
    } else {
      return val.device_cuda_mem().device_id();
    }
  }
};
}  // namespace std

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_
