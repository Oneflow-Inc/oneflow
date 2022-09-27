#ifndef ONEFLOW_CORE_COMMON_MEM_UTIL_H_
#define ONEFLOW_CORE_COMMON_MEM_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/mem_util.h"

#include <chrono>
#include <sstream>
#include <string>

namespace oneflow {

void ProcessMemUsage(double& vm_usage, double& resident_set);
}  // namespace oneflow

#define LOG_MEM(...)                                                                        \
  double vm_=0, rss_=0;                                                                     \
  ProcessMemUsage(vm_, rss_);                                                               \
  VLOG(1) << "File " __FILE__  << ", Line " <<  __LINE__ << ", Func "                       \
          << __FUNCTION__ << ", Mem size vm " << vm_ << "MB, rss " << rss_ << " MB"

#endif  // ONEFLOW_CORE_COMMON_MEM_UTIL_H_