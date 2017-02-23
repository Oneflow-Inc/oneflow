#ifndef ONEFLOW_COMMON_UTIL_H
#define ONEFLOW_COMMON_UTIL_H

#include <unordered_set>

namespace oneflow {

#define DISALLOW_COPY(ClassName) \
  ClassName(const ClassName&) = delete; \
  ClassName& operator = (const ClassName&) = delete;

#define DISALLOW_MOVE(ClassName) \
  ClassName(ClassName&&) = delete; \
  ClassName& operator = (ClassName&&) = delete;

#define DISALLOW_COPY_AND_MOVE(ClassName) \
  DISALLOW_COPY(ClassName) \
  DISALLOW_MOVE(ClassName)

// we have to implement it to hack the glog/gtest macro
template<typename T>
bool IsEqual(const std::unordered_set<T>& lhs,
             const std::unordered_set<T>& rhs) {
  return lhs == rhs;
}

enum class FloatType {
  kFloat,
  kDouble
};

inline size_t GetFloatByteSize(FloatType ft) {
  if (ft == FloatType::kFloat) {
    return 4;
  } else {
    return 8;
  }
}

} // namespace oneflow

#endif // ONEFLOW_COMMON_UTIL_H
