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
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include <cfenv>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/platform.h"
#include <csignal>
#include <limits>

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

namespace oneflow {

#define DEFINE_ONEFLOW_STR2INT_CAST(dst_type, cast_func) \
  template<>                                             \
  dst_type oneflow_cast(const std::string& s) {          \
    char* end_ptr = nullptr;                             \
    dst_type ret = cast_func(s.c_str(), &end_ptr, 0);    \
    CHECK_EQ(*end_ptr, '\0');                            \
    return ret;                                          \
  }

DEFINE_ONEFLOW_STR2INT_CAST(long, strtol);
DEFINE_ONEFLOW_STR2INT_CAST(unsigned long, strtoul);
DEFINE_ONEFLOW_STR2INT_CAST(long long, strtoll);
DEFINE_ONEFLOW_STR2INT_CAST(unsigned long long, strtoull);

DEFINE_ONEFLOW_STR2INT_CAST(signed char, strtol);
DEFINE_ONEFLOW_STR2INT_CAST(short, strtol);
DEFINE_ONEFLOW_STR2INT_CAST(int, strtol);

DEFINE_ONEFLOW_STR2INT_CAST(unsigned char, strtoul);
DEFINE_ONEFLOW_STR2INT_CAST(unsigned short, strtoul);
DEFINE_ONEFLOW_STR2INT_CAST(unsigned int, strtoul);

template<>
float oneflow_cast(const std::string& s) {
  char* end_ptr = nullptr;
  float ret = strtof(s.c_str(), &end_ptr);
  CHECK_EQ(*end_ptr, '\0');
  return ret;
}

template<>
double oneflow_cast(const std::string& s) {
  char* end_ptr = nullptr;
  double ret = strtod(s.c_str(), &end_ptr);
  CHECK_EQ(*end_ptr, '\0');
  return ret;
}

#ifdef OF_PLATFORM_POSIX
// COMMAND(feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW));
#endif

void AbortSignalHandler(int signal) { exit(-1); }

COMMAND(std::signal(SIGINT, AbortSignalHandler));

size_t GetAvailableCpuMemSize() {
#if defined(__linux__)
  std::ifstream mem_info("/proc/meminfo");
  CHECK(mem_info.good()) << "can't open file: /proc/meminfo";
  std::string line;
  while (std::getline(mem_info, line).good()) {
    std::string token;
    const char* p = line.c_str();
    p = StrToToken(p, " ", &token);
    if (token != "MemAvailable:") { continue; }
    CHECK_NE(*p, '\0');
    p = StrToToken(p, " ", &token);
    size_t mem_available = oneflow_cast<size_t>(token);
    CHECK_NE(*p, '\0');
    p = StrToToken(p, " ", &token);
    CHECK_EQ(token, "kB");
    return mem_available * 1024;
  }
  LOG(FATAL) << "can't find MemAvailable in /proc/meminfo";
#elif defined(__APPLE__)
  // macOS will eagerly make use of all memory so there is no point querying it
  return std::numeric_limits<size_t>::max();
#else
  return 0;
#endif
}

bool IsKernelSafeInt32(int64_t n) { return n <= GetMaxVal<int32_t>() / 2; }

}  // namespace oneflow
