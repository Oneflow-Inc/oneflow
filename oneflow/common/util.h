#ifndef ONEFLOW_COMMON_UTIL_H
#define ONEFLOW_COMMON_UTIL_H

#include <set>

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

template <typename T>
bool is_equal(const std::set<T>& lhs, const std::set<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (const T& lhs_elem : lhs) {
    if (rhs.find(lhs_elem) == rhs.end()) {
      return false;
    }
  }
  return true;
}

} // namespace oneflow

#endif // ONEFLOW_COMMON_UTIL_H
