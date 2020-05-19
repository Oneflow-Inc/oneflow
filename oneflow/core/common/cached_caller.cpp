#include "oneflow/core/common/util.h"
#include "oneflow/core/common/cached_caller.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

bool IsThreadLocalCacheEnabled() {
  if (Global<ResourceDesc>::Get() == nullptr) { return true; }
  return Global<ResourceDesc>::Get()->enable_thread_local_cache();
}

}  // namespace oneflow
