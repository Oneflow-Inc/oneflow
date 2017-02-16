#ifndef ONEFLOW_COMMON_UTIL_H
#define ONEFLOW_COMMON_UTIL_H

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

} // namespace oneflow

#endif // ONEFLOW_COMMON_UTIL_H
