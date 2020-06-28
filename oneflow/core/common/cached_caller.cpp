#include "oneflow/core/common/util.h"
#include "oneflow/core/common/cached_caller.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

bool IsThreadLocalCacheEnabled() {
  if (Global<ResourceDesc, ForSession>::Get() == nullptr) { return true; }
  return Global<ResourceDesc, ForSession>::Get()->enable_thread_local_cache();
}

}  // namespace oneflow
