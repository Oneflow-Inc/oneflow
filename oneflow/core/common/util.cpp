#include "oneflow/core/common/util.h"
#include <cfenv>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/platform.h"

#ifdef PLATFORM_POSIX
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

#ifdef PLATFORM_POSIX
COMMAND(feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW));
#endif

void RedirectStdoutAndStderrToGlogDir() {
  PCHECK(freopen(JoinPath(LogDir(), "stdout").c_str(), "a+", stdout));
  PCHECK(freopen(JoinPath(LogDir(), "stderr").c_str(), "a+", stderr));
}

void CloseStdoutAndStderr() {
  PCHECK(fclose(stdout) == 0);
  PCHECK(fclose(stderr) == 0);
}

size_t GetAvailableCpuMemSize() {
#ifdef PLATFORM_POSIX
  struct sysinfo sys_info;
  PCHECK(sysinfo(&sys_info) == 0);
  return sys_info.freeram * sys_info.mem_unit;
#else
  return 0;  // TODO
#endif
}

}  // namespace oneflow
