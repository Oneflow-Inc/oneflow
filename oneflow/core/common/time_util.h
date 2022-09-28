/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMMON_TIME_UTIL_H_
#define ONEFLOW_CORE_COMMON_TIME_UTIL_H_

#include <chrono>
#include <sstream>
#include <string>

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/mem_util.h"

namespace oneflow {

template<typename DurationT>
struct Duration {
  static const std::string& Repr() {
    static const std::string repr = "";
    return repr;
  }
};

#define DEFINE_DURATION_TRAIT(time_type)             \
  template<>                                         \
  struct Duration<typename std::chrono::time_type> { \
    static const std::string& Repr() {               \
      static const std::string repr = #time_type;    \
      return repr;                                   \
    }                                                \
  };

DEFINE_DURATION_TRAIT(nanoseconds)
DEFINE_DURATION_TRAIT(microseconds)
DEFINE_DURATION_TRAIT(milliseconds)
DEFINE_DURATION_TRAIT(seconds)
DEFINE_DURATION_TRAIT(minutes)
DEFINE_DURATION_TRAIT(hours)
#undef DEFINE_DURATION_TRAIT

template<class Resolution = std::chrono::seconds>
class TimeCounter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TimeCounter);
  explicit TimeCounter(bool with_log = true, bool with_mem = false)
      : with_log_(with_log), with_mem_(with_mem) {}
  ~TimeCounter() = default;

  void Count(const std::string& log_prefix = "", int v_log_level = 0);

 private:
  using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                   std::chrono::high_resolution_clock, std::chrono::steady_clock>;

  Clock::time_point start_{Clock::now()};
  bool with_log_{false};
  bool with_mem_{false};
};

template<class Resolution>
void TimeCounter<Resolution>::Count(const std::string& log_prefix, int v_log_level) {
  const auto end = Clock::now();
  auto dur = std::chrono::duration_cast<Resolution>(end - start_).count();
  if (with_log_ && v_log_level >= 0) {
    std::ostringstream oss;
    oss << log_prefix << " time elapsed: " << std::to_string(dur) << " "
        << Duration<Resolution>::Repr();
    if (with_mem_) {
      double vm = 0, rss = 0;
      ProcessMemUsage(vm, rss);
      oss << ", Mem size RSS " << rss << " MB, VM " << vm << " MB";
    }
    if (v_log_level == 0) {
      LOG(INFO) << oss.str() << ".";
    } else {
      VLOG(v_log_level) << oss.str() << ".";
    }
  }
  start_ = end;
  return;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TIME_UTIL_H_
