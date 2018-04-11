#ifndef ONEFLOW_CORE_COMMON_UTIL_H_
#define ONEFLOW_CORE_COMMON_UTIL_H_

#include "oneflow/core/common/preprocessor.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <forward_list>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

DECLARE_string(log_dir);

namespace oneflow {

#define OF_DISALLOW_COPY(ClassName)     \
  ClassName(const ClassName&) = delete; \
  ClassName& operator=(const ClassName&) = delete;

#define OF_DISALLOW_MOVE(ClassName) \
  ClassName(ClassName&&) = delete;  \
  ClassName& operator=(ClassName&&) = delete;

#define OF_DISALLOW_COPY_AND_MOVE(ClassName) \
  OF_DISALLOW_COPY(ClassName)                \
  OF_DISALLOW_MOVE(ClassName)

#define UNIMPLEMENTED() LOG(FATAL) << "UNIMPLEMENTED"

#define TODO() LOG(FATAL) << "TODO";

template<typename T>
class Global final {
 public:
  static T* Get() { return *GetPPtr(); }
  static void SetAllocated(T* val) { *GetPPtr() = val; }
  template<typename... Args>
  static void New(Args&&... args) {
    CHECK(Get() == nullptr);
    LOG(INFO) << "NewGlobal " << typeid(T).name();
    *GetPPtr() = new T(std::forward<Args>(args)...);
  }
  static void Delete() {
    if (Get() != nullptr) {
      LOG(INFO) << "DeleteGlobal " << typeid(T).name();
      delete Get();
      *GetPPtr() = nullptr;
    }
  }

 private:
  static T** GetPPtr() {
    static T* ptr = nullptr;
    return &ptr;
  }
};

#define COMMAND(...)                                                \
  namespace {                                                       \
  struct OF_PP_CAT(CommandT, __LINE__) {                            \
    OF_PP_CAT(CommandT, __LINE__)() { __VA_ARGS__; }                \
  };                                                                \
  OF_PP_CAT(CommandT, __LINE__) OF_PP_CAT(g_command_var, __LINE__); \
  }

template<typename T>
bool operator==(const std::weak_ptr<T>& lhs, const std::weak_ptr<T>& rhs) {
  return lhs.lock().get() == rhs.lock().get();
}

template<typename Key, typename T, typename Hash = std::hash<Key>>
using HashMap = std::unordered_map<Key, T, Hash>;

template<typename Key, typename Hash = std::hash<Key>>
using HashSet = std::unordered_set<Key, Hash>;

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

inline std::string NewUniqueId() {
  static int64_t id = 0;
  return std::to_string(id++);
}

inline const std::string& LogDir() {
  static std::string v = FLAGS_log_dir;
  return v;
}

template<typename K, typename V>
void EraseIf(HashMap<K, V>* hash_map,
             std::function<bool(typename HashMap<K, V>::iterator)> cond) {
  for (auto it = hash_map->begin(); it != hash_map->end();) {
    if (cond(it)) {
      hash_map->erase(it++);
    } else {
      ++it;
    }
  }
}

#define OF_DECLARE_ENUM_TO_OSTREAM_FUNC(EnumType) \
  std::ostream& operator<<(std::ostream& out_stream, const EnumType&)

#define OF_DEFINE_ENUM_TO_OSTREAM_FUNC(EnumType)                          \
  std::ostream& operator<<(std::ostream& out_stream, const EnumType& x) { \
    out_stream << static_cast<int>(x);                                    \
    return out_stream;                                                    \
  }

template<typename OutType, typename InType>
OutType oneflow_cast(const InType&);

inline uint32_t NewRandomSeed() {
  static std::mt19937 gen{std::random_device{}()};
  return gen();
}

// Work around the following issue on Windows
// https://stackoverflow.com/questions/33218522/cuda-host-device-variables
// const float LOG_THRESHOLD = 1e-20;
#define LOG_THRESHOLD (1e-20)
#define MAX_WITH_LOG_THRESHOLD(x) ((x) > LOG_THRESHOLD ? (x) : LOG_THRESHOLD)
#define SAFE_LOG(x) logf(MAX_WITH_LOG_THRESHOLD(x))

#if defined(WITH_CUDA)
#define DEVICE_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU) \
  OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU)
#else
#define DEVICE_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU)
#endif

#define DIM_SEQ (1)(2)(3)(4)(5)(6)(7)(8)

#define BOOL_SEQ (true)(false)
#define PARALLEL_POLICY_SEQ \
  (ParallelPolicy::kModelParallel)(ParallelPolicy::kDataParallel)
#define ENCODE_CASE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(EncodeCase::kRaw) \
  OF_PP_MAKE_TUPLE_SEQ(EncodeCase::kJpeg)

#define FOR_RANGE(type, i, begin, end) for (type i = begin; i < end; ++i)
#define FOR_EACH(it, container) \
  for (auto it = container.begin(); it != container.end(); ++it)

void RedirectStdoutAndStderrToGlogDir();
void CloseStdoutAndStderr();

inline double GetCurTime() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

size_t GetAvailableCpuMemSize();

template<typename T>
void Erase(
    T& container, std::function<bool(const typename T::value_type&)> NeedErase,
    std::function<void(const typename T::value_type&)> EraseElementHandler) {
  auto iter = container.begin();
  auto erase_from = container.end();
  while (iter != erase_from) {
    if (NeedErase(*iter)) {
      --erase_from;
      if (iter == erase_from) { break; }
      std::swap(*iter, *erase_from);
    } else {
      ++iter;
    }
  }
  for (; iter != container.end(); ++iter) { EraseElementHandler(*iter); }
  if (erase_from != container.end()) {
    container.erase(erase_from, container.end());
  }
}

template<typename T>
void Erase(T& container,
           std::function<bool(const typename T::value_type&)> NeedErase) {
  Erase<T>(container, NeedErase, [](const typename T::value_type&) {});
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_UTIL_H_
