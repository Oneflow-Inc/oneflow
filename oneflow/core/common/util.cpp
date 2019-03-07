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

#define DEFINE_ONEFLOW_ARITHMETIC_CAST(in_type_pair, out_type_pair) \
  template<>                                                        \
  OF_PP_PAIR_FIRST(out_type_pair)                                   \
  oneflow_cast(const OF_PP_PAIR_FIRST(in_type_pair) & s) {          \
    return static_cast<OF_PP_PAIR_FIRST(out_type_pair)>(s);         \
  }

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(DEFINE_ONEFLOW_ARITHMETIC_CAST, ARITHMETIC_DATA_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

#define DEFINE_ONEFLOW_HALF2ARITHMETIC_CAST(out_type, out_type_enum) \
  template<>                                                         \
  out_type oneflow_cast(const float16& s) {                          \
    return half_float::half_cast<out_type>(s);                       \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_ONEFLOW_HALF2ARITHMETIC_CAST, ARITHMETIC_DATA_TYPE_SEQ)

#define DEFINE_ONEFLOW_ARITHMETIC2HALF_CAST(in_type, in_type_enum) \
  template<>                                                       \
  float16 oneflow_cast(const in_type& s) {                         \
    return half_float::half_cast<float16>(s);                      \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_ONEFLOW_ARITHMETIC2HALF_CAST, ARITHMETIC_DATA_TYPE_SEQ)

#ifdef PLATFORM_POSIX
COMMAND(feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW));
#endif

void RedirectStdoutAndStderrToGlogDir() {
  PCHECK(freopen(JoinPath(FLAGS_log_dir, "stdout").c_str(), "a+", stdout));
  PCHECK(freopen(JoinPath(FLAGS_log_dir, "stderr").c_str(), "a+", stderr));
}

void CloseStdoutAndStderr() {
  PCHECK(fclose(stdout) == 0);
  PCHECK(fclose(stderr) == 0);
}

size_t GetAvailableCpuMemSize() {
#ifdef PLATFORM_POSIX
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
#else
  TODO();
#endif
  return 0;
}

std::string LogDir() {
  char hostname[255];
  CHECK_EQ(gethostname(hostname, sizeof(hostname)), 0);
  std::string v = FLAGS_log_dir + "/" + std::string(hostname);
  return v;
}

}  // namespace oneflow
