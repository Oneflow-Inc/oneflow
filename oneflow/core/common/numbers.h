#ifndef ONEFLOW_CORE_COMMON_NUMBERS_H_
#define ONEFLOW_CORE_COMMON_NUMBERS_H_

#include "tensorflow/core/lib/strings/numbers.h"

namespace oneflow {

inline uint64_t Stou64OrDie(const std::string& s) {
  tensorflow::uint64 ret = 0;
  CHECK(tensorflow::strings::safe_strtou64(s, &ret));
  return ret;
}

inline int64_t Sto64OrDie(const std::string& s) {
  tensorflow::int64 ret = 0;
  CHECK(tensorflow::strings::safe_strto64(s, &ret));
  return ret;
}

inline uint32_t Stou32OrDie(const std::string& s) {
  uint32_t ret = 0;
  CHECK(tensorflow::strings::safe_strtou32(s, &ret));
  return ret;
}

inline int32_t Sto32orDie(const std::string& s) {
  int32_t ret = 0;
  CHECK(tensorflow::strings::safe_strto32(s, &ret));
  return ret;
}

}

#endif  // ONEFLOW_CORE_COMMON_NUMBERS_H_ 
