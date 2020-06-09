#ifndef OF_TIME_UTIL_H_
#define OF_TIME_UTIL_H_

#include "oneflow/core/framework/framework.h"
namespace oneflow {

namespace envtime {

static constexpr uint64_t kMicrosToNanos = 1000ULL;
static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

inline uint64_t NowNanos() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos + static_cast<uint64_t>(ts.tv_nsec));
}

inline double GetWallTime() { return static_cast<double>(NowNanos() / kMicrosToNanos) / 1.0e6; }

}  // namespace envtime
}  // namespace oneflow

#endif