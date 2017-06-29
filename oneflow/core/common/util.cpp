#include "oneflow/core/common/util.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace oneflow {

template<>
int32_t oneflow_cast(const std::string& s) {
  int32_t ret = 0;
  CHECK(tensorflow::strings::safe_strto32(s, &ret));
  return ret;
}

template<>
uint32_t oneflow_cast(const std::string& s) {
  uint32_t ret = 0;
  CHECK(tensorflow::strings::safe_strtou32(s, &ret));
  return ret;
}

template<>
int64_t oneflow_cast(const std::string& s) {
  tensorflow::int64 ret = 0;
  CHECK(tensorflow::strings::safe_strto64(s, &ret));
  return ret;
}

template<>
uint64_t oneflow_cast(const std::string& s) {
  tensorflow::uint64 ret = 0;
  CHECK(tensorflow::strings::safe_strtou64(s, &ret));
  return ret;
}

} // namespace oneflow
