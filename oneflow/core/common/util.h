#ifndef ONEFLOW_CORE_COMMON_UTIL_H_
#define ONEFLOW_CORE_COMMON_UTIL_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
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
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace oneflow {

#define INLINE inline

#define OF_DISALLOW_COPY(ClassName)     \
  ClassName(const ClassName&) = delete; \
  ClassName& operator=(const ClassName&) = delete;

#define OF_DISALLOW_MOVE(ClassName) \
  ClassName(ClassName&&) = delete;  \
  ClassName& operator=(ClassName&&) = delete;

#define OF_DISALLOW_COPY_AND_MOVE(ClassName) \
  OF_DISALLOW_COPY(ClassName)                \
  OF_DISALLOW_MOVE(ClassName)

#define UNEXPECTED_RUN() LOG(FATAL) << "Unexpected Run";

#define TODO() LOG(FATAL) << "TODO";

#define OF_SINGLETON(ClassName)            \
  static ClassName* Singleton() {          \
    static ClassName* ptr = new ClassName; \
    return ptr;                            \
  }

#define COMMAND(...)            \
  namespace {                   \
  struct CommandT {             \
    CommandT() { __VA_ARGS__; } \
  };                            \
  CommandT g_command_var;       \
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

inline std::string LogDir() {
  static std::string log_dir = std::getenv("GLOG_log_dir");
  return log_dir;
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

template<template<class, class, class...> class C, typename K, typename V,
         typename... Args>
V GetOrDefault(const C<K, V, Args...>& m, K const& key, const V& defval) {
  typename C<K, V, Args...>::const_iterator it = m.find(key);
  if (it == m.end()) {
    return defval;
  } else {
    return it->second;
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

inline std::string GetClassName(const std::string& prettyFunction) {
  size_t colons = prettyFunction.rfind("::");
  if (colons == std::string::npos) return "::";
  size_t begin = prettyFunction.substr(0, colons).rfind("::") + 2;
  size_t end = colons - begin;

  return prettyFunction.substr(begin, end);
}

#ifdef _MSC_VER
#define __CLASS_NAME__ GetClassName(__FUNCSIG__)
#else
#define __CLASS_NAME__ GetClassName(__PRETTY_FUNCTION__)
#endif

#define DEVICE_TYPE_SEQ (DeviceType::kCPU)(DeviceType::kGPU)
#define BOOL_SEQ (true)(false)
#define PARALLEL_POLICY_SEQ \
  (ParallelPolicy::kModelParallel)(ParallelPolicy::kDataParallel)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_UTIL_H_
