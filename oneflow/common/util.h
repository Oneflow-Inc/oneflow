#ifndef ONEFLOW_COMMON_UTIL_H
#define ONEFLOW_COMMON_UTIL_H

#include <unordered_set>
#include <unordered_map>
#include <algorithm>
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

template<typename Key, typename T, typename Hash = std::hash<Key>>
using HashMap = std::unordered_map<Key, T, Hash>;

template<typename T, typename... Args>
std::unique_ptr<T> of_make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template<typename T>
void SortAndRemoveDuplication(std::vector<T>* vec) {
  std::sort(vec->begin(), vec->end());
  auto unique_it = std::unique(vec->begin(), vec->end());
  vec->erase(unique_it, vec->end());
}

inline unsigned long long StoullOrDie(const std::string& s) {
  unsigned long long ret = 0;
  try {
    ret = std::stoull(s);
  } catch (std::exception& e){
    LOG(FATAL) << "Error: " << s;
  }
  return ret;
}

inline std::string NewUniqueId() {
  static uint64_t id = 0;
  return std::to_string(id++);
}

} // namespace oneflow

#endif // ONEFLOW_COMMON_UTIL_H
