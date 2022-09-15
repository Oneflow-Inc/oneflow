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

#include "oneflow/core/common/util.h"

#include <unistd.h>
#include <sys/sysinfo.h>
#include <chrono>
#include <sstream>
#include <string>

namespace oneflow {


namespace {
void process_mem_usage(double& vm_usage, double& resident_set)
{
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0.0;
   resident_set = 0.0;

   // 'file' stat seems to give the most reliable results
   //
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   //
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
   //
   unsigned long vsize = 0;
   long rss = 0;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   stat_stream.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) >> 20; // in case x86-64 is configured to use 2MB pages
   vm_usage     = vsize >> 20;
   resident_set = rss * page_size_kb;
}
}  // namespace

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
  explicit TimeCounter(bool with_log = false) : with_log_(with_log) {}
  ~TimeCounter() = default;

  inline void Reset() { start_ = Clock::now(); }

  double Count(const std::string& log_prefix = "", int v_log_level = 0);
 private:
  using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                   std::chrono::high_resolution_clock, std::chrono::steady_clock>;

  Clock::time_point start_{Clock::now()};
  bool with_log_{false};
};

template<class Resolution>
double TimeCounter<Resolution>::Count(const std::string& log_prefix, int v_log_level) {
  const auto end = Clock::now();
  auto dur = std::chrono::duration_cast<Resolution>(end - start_).count();
  if (with_log_ && v_log_level >= 0) {
    std::ostringstream oss;
    oss << log_prefix << " time elapsed: " << std::to_string(dur) << " "
        << Duration<Resolution>::Repr();
    double vm=0, rss=0;
    process_mem_usage(vm, rss);
    if (v_log_level == 0) {
      LOG(INFO) << oss.str() << ", mem size vm " << vm << " rss " << rss << " MB";
    } else {
      VLOG(v_log_level) << oss.str() << ", mem size vm " << vm << " rss " << rss << " MB";
    }
  }
  start_ = end;
  double time_cnt = static_cast<double>(dur);
  return time_cnt;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TIME_UTIL_H_