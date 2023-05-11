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
#include <unistd.h>
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

// If the interrupt during object malloc is changed to exit, the exit function indicates a normal
// exit, triggering the object destructor function and then triggering object free. Since there is a
// lock in malloc, if malloc and free obtain the same lock, it can cause a deadlock, which prevents
// the process from exiting. After calling abort, the OS forces the program to exit,
// relying on the OS to do resource cleanup, which can avoid the deadlock issue.
// Process inability to exit can be more troublesome than potential resource leaks. If we find that
// abort causes unreleased resources later, we can use exit in a local scope rather than globally.
// Reference: https://github.com/Oneflow-Inc/OneTeam/issues/1954
void AbortSignalHandler(int signal) { std::abort(); }
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
  return sysconf(_SC_PAGESIZE) * sysconf(_SC_AVPHYS_PAGES);
#elif defined(__APPLE__)
  // macOS will eagerly make use of all memory so there is no point querying it
  return std::numeric_limits<size_t>::max();
#else
  UNIMPLEMENTED();
  return 0;
#endif
}

bool IsKernelSafeInt32(int64_t n) { return n <= GetMaxVal<int32_t>() / 2; }

namespace {

bool CaseInsensitiveStringEquals(const std::string& lhs, const std::string& rhs) {
  return lhs.size() == rhs.size()
         && std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                       [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

bool StringToBool(const std::string& str) {
  return CaseInsensitiveStringEquals(str, "1") || CaseInsensitiveStringEquals(str, "true")
         || CaseInsensitiveStringEquals(str, "yes") || CaseInsensitiveStringEquals(str, "on")
         || CaseInsensitiveStringEquals(str, "y");
}

bool StringToInteger(const std::string& str, int64_t* value) {
  char* end;
  int64_t v = std::strtoll(str.data(), &end, 10);
  if (end == str.data()) {
    return false;
  } else {
    *value = v;
    return true;
  }
}

bool StringToFloat(const std::string& str, double* value) {
  char* end = nullptr;
  double v = std::strtof(str.data(), &end);
  if (end == str.data()) {
    return false;
  } else {
    *value = v;
    return true;
  }
}

}  // namespace

bool ParseBooleanFromEnv(const std::string& env_var, bool default_value) {
  const char* env_p = std::getenv(env_var.c_str());
  if (env_p == nullptr) {
    return default_value;
  } else {
    return StringToBool(env_p);
  }
}

int64_t ParseIntegerFromEnv(const std::string& env_var, int64_t default_value) {
  const char* env_p = std::getenv(env_var.c_str());
  if (env_p == nullptr) { return default_value; }
  int64_t value;
  if (StringToInteger(env_p, &value)) {
    return value;
  } else {
    return default_value;
  }
}

double ParseFloatFromEnv(const std::string& env_var, double default_value) {
  const char* env_p = std::getenv(env_var.c_str());
  if (env_p == nullptr) { return default_value; }
  double value = default_value;
  StringToFloat(env_p, &value);
  return value;
}

std::string GetStringFromEnv(const std::string& env_var, const std::string& default_value) {
  const char* env_p = std::getenv(env_var.c_str());
  if (env_p == nullptr) {
    return default_value;
  } else {
    return env_p;
  }
}

}  // namespace oneflow
