#ifndef OF_TIME_UTIL_H_
#define OF_TIME_UTIL_H_

#include "oneflow/core/framework/framework.h"
namespace oneflow {

namespace envtime {

static constexpr uint64_t kMicroTimeToNanoTime = 1000ULL;
static constexpr uint64_t kSecondToNanoTime = 1000ULL * 1000ULL * 1000ULL;
static constexpr uint64_t kMircoTimeToSecondTime = 1000ULL * 1000ULL;

inline uint64_t CurrentNanoTime() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<uint64_t>(ts.tv_sec) * kSecondToNanoTime
          + static_cast<uint64_t>(ts.tv_nsec));
}

inline uint64_t CurrentMircoTime() { return CurrentNanoTime() / kMicroTimeToNanoTime; }

inline uint64_t CurrentSecondTime() { return CurrentMircoTime() / kMircoTimeToSecondTime; }

inline double GetWallTime() {
  return static_cast<double>(CurrentNanoTime() / kMicroTimeToNanoTime) / 1.0e6;
}

}  // namespace envtime
}  // namespace oneflow

#endif