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

#include "nlohmann/json.hpp"

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/mem_util.h"
#include "oneflow/core/job/utils/progress_bar.h"

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
class CostCounter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CostCounter);
  explicit CostCounter(bool with_log = true, bool with_mem = false)
      : with_log_(with_log), with_mem_(with_mem) {}
  ~CostCounter() = default;

  void Count(const std::string& log_prefix = "", int v_log_level = 0, bool log_progress = false);

 private:
  using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                   std::chrono::high_resolution_clock, std::chrono::steady_clock>;

  Clock::time_point start_{Clock::now()};
  bool with_log_{false};
  bool with_mem_{false};
};

template<class Resolution>
void CostCounter<Resolution>::Count(const std::string& log_prefix, int v_log_level,
                                    bool log_progress) {
  if (log_progress) { CHECK_JUST(LogProgress(log_prefix)); }

  const auto end = Clock::now();
  if (FLAGS_minloglevel <= 0 && VLOG_IS_ON(v_log_level) && with_log_ && v_log_level >= 0) {
    // only do time/mem count and log when glog level is INFO and VLOG level is matched.
    auto dur = std::chrono::duration_cast<Resolution>(end - start_).count();

    nlohmann::json json_log;
    json_log["loc"] = log_prefix;
    json_log["time_cost"] = std::to_string(dur) + " " + Duration<Resolution>::Repr();

    if (with_mem_) {
#ifdef __linux__
      double vm = 0, rss = 0;
      ProcessMemUsage(&vm, &rss);
      json_log["mem_rss"] = std::to_string(rss) + " MB";
#endif  // __linux__
    }

    if (v_log_level == 0) {
      LOG(INFO) << "[count log]" << json_log.dump();
    } else {
      VLOG(v_log_level) << "[count log]" << json_log.dump();
    }
  }
  start_ = end;
  return;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TIME_UTIL_H_
