#ifndef ONEFLOW_CORE_COMMON_TO_STRING_H_
#define ONEFLOW_CORE_COMMON_TO_STRING_H_

#include <string>

namespace oneflow {

template<typename T>
inline std::string ToString(const T& value) {
  return std::to_string(value);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TO_STRING_H_
