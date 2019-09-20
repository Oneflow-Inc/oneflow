#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

bool MemoryCaseUtil::GetCommonMemoryCase(const MemoryCase& a, const MemoryCase& b,
                                         MemoryCase* common) {
  if (a.has_device_cuda_mem() && b.has_device_cuda_mem()) {
    if (a.device_cuda_mem().device_id() == b.device_cuda_mem().device_id()) {
      *common = a;
      return true;
    } else {
      return false;
    }
  } else if (a.has_host_mem() && b.has_host_mem()) {
    *common = a;
    if (b.host_mem().has_cuda_pinned_mem()) {
      *common->mutable_host_mem()->mutable_cuda_pinned_mem() = b.host_mem().cuda_pinned_mem();
    }
    if (b.host_mem().has_used_by_network()) {
      common->mutable_host_mem()->set_used_by_network(true);
    }
    return true;
  } else {
    return false;
  }
}

MemoryCase MemoryCaseUtil::GetHostPinnedMemoryCaseForRegstSeparatedHeader(
    const MemoryCase& mem_case) {
  CHECK(mem_case.has_device_cuda_mem());
  MemoryCase ret;
  ret.mutable_host_mem()->mutable_cuda_pinned_mem()->set_device_id(
      mem_case.device_cuda_mem().device_id());
  return ret;
}

}  // namespace oneflow
