#ifndef ONEFLOW_CORE_FRAMEWORK_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_UTIL_H_

#include "oneflow/core/common/util.h"

namespace std {

template<>
struct hash<std::pair<std::string, int32_t>> {
  std::size_t operator()(const std::pair<std::string, int32_t>& p) const {
    return std::hash<std::string>{}(p.first) ^ std::hash<int32_t>{}(p.second);
  }
};

}  // namespace std

namespace oneflow {

namespace user_op {}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_UTIL_H_
