#ifndef ONEFLOW_CORE_COMMON_MEM_UTIL_H_
#define ONEFLOW_CORE_COMMON_MEM_UTIL_H_

#include <chrono>
#include <sstream>
#include <string>

#include "oneflow/core/common/util.h"

namespace oneflow {
void ProcessMemUsage(double& vm_usage, double& resident_set);
}  // namespace oneflow

#define LOG_MEM(...)                                                                \
  double vm_ = 0, rss_ = 0;                                                         \
  ProcessMemUsage(vm_, rss_);                                                       \
  VLOG(1) << "File " __FILE__ << ", Line " << __LINE__ << ", Func " << __FUNCTION__ \
          << ", Mem size RSS " << rss_ << "MB, VM " << vm_ << " MB."

#endif  // ONEFLOW_CORE_COMMON_MEM_UTIL_H_
