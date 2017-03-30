#ifndef ONEFLOW_COMMON_UTIL_H
#define ONEFLOW_COMMON_UTIL_H

#include <unordered_set>
#include "glog/logging.h"
#include "google/protobuf/message.h"
#include "google/protobuf/descriptor.h"

namespace oneflow {

#define OF_DISALLOW_COPY(ClassName) \
  ClassName(const ClassName&) = delete; \
  ClassName& operator = (const ClassName&) = delete;

#define OF_DISALLOW_MOVE(ClassName) \
  ClassName(ClassName&&) = delete; \
  ClassName& operator = (ClassName&&) = delete;

#define OF_DISALLOW_COPY_AND_MOVE(ClassName) \
  OF_DISALLOW_COPY(ClassName) \
  OF_DISALLOW_MOVE(ClassName)

#define UNEXPECTED_RUN() \
  LOG(FATAL) << "Unexpected Run";

#define TODO() \
  LOG(FATAL) << "TODO";

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
