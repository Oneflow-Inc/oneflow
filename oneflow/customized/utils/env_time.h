#ifndef OF_TIME_UTIL_H_
#define OF_TIME_UTIL_H_

#include "oneflow/core/framework/framework.h"
namespace oneflow {

namespace envtime {

static constexpr uint64_t kMicroTimeToNanoTime = 1000ULL;
static constexpr uint64_t kSecondsToNanoTime = 1000ULL * 1000ULL * 1000ULL;

inline uint64_t CurrentNanoTime() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanoTime
          + static_cast<uint64_t>(ts.tv_nsec));
}

inline uint64_t CurrentMircoTime() { return CurrentNanoTime() / kMicroTimeToNanoTime; }

inline double GetWallTime() {
  return static_cast<double>(CurrentNanoTime() / kMicroTimeToNanoTime) / 1.0e6;
}

}  // namespace envtime
}  // namespace oneflow

#endif