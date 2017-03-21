#ifndef ONEFLOW_COMMON_UTIL_H
#define ONEFLOW_COMMON_UTIL_H

#include <unordered_set>
#include "glog/logging.h"
#include "google/protobuf/message.h"
#include "google/protobuf/descriptor.h"

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
    return sizeof(float);
  } else {
    return sizeof(double);
  }
}

template<typename Target, typename Source>
inline Target of_dynamic_cast(Source arg) {
  Target ret = dynamic_cast<Target> (arg);
  CHECK_NOTNULL(ret);
  return ret;
}

inline bool operator == (const google::protobuf::MessageLite& lhs,
                         const google::protobuf::MessageLite& rhs) {
  return lhs.SerializeAsString() == rhs.SerializeAsString();
}

inline bool operator != (const google::protobuf::MessageLite& lhs,
                         const google::protobuf::MessageLite& rhs) {
  return !(lhs == rhs);
}

} // namespace oneflow

#endif // ONEFLOW_COMMON_UTIL_H
