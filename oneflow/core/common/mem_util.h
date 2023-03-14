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
#ifndef ONEFLOW_CORE_COMMON_MEM_UTIL_H_
#define ONEFLOW_CORE_COMMON_MEM_UTIL_H_

#include <chrono>
#include <sstream>
#include <string>

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {
void ProcessMemUsage(double* vm_usage, double* resident_set);
std::string FormatMemSize(uint64_t size);
Maybe<double> GetCPUMemoryUsed();
}  // namespace oneflow

#define LOG_MEM(...)                                                                \
  double vm_ = 0, rss_ = 0;                                                         \
  ProcessMemUsage(&vm_, &rss_);                                                     \
  VLOG(1) << "File " __FILE__ << ", Line " << __LINE__ << ", Func " << __FUNCTION__ \
          << ", Mem size RSS " << rss_ << "MB."

#endif  // ONEFLOW_CORE_COMMON_MEM_UTIL_H_
