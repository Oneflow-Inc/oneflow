#include <algorithm>
#include "oneflow/core/kernel/util/thread_name.h"

#if defined(__GLIBC__) && !defined(__APPLE__) && !defined(__ANDROID__)
#define C10_HAS_PTHREAD_SETNAME_NP
#endif

#ifdef C10_HAS_PTHREAD_SETNAME_NP
#include <pthread.h>
#endif

namespace oneflow {
namespace internal {

void setThreadName(std::string name) {
#ifdef C10_HAS_PTHREAD_SETNAME_NP
  constexpr size_t kMaxThreadName = 15;
  name.resize(std::min(name.size(), kMaxThreadName));

  pthread_setname_np(pthread_self(), name.c_str());
#endif
}

void NUMABind(int numa_node_id) {} // not enable numa by default

} // namespace internal
} // namespace oneflow
