#ifndef ONEFLOW_CORE_SCHEDULE_UTILS_UTILS_H_
#define ONEFLOW_CORE_SCHEDULE_UTILS_UTILS_H_

namespace oneflow {
namespace schedule {

#define MACRO_CONCAT_(a, b) a##b
#define MACRO_CONCAT(a, b) MACRO_CONCAT_(a, b)

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

template<typename T, typename... Args>
inline std::unique_ptr<T> unique_ptr_new(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template<typename T, typename... Args>
inline std::shared_ptr<T> shared_ptr_new(Args&&... args) {
  return std::shared_ptr<T>(new T(std::forward<Args>(args)...));
}

inline uint64_t GetAutoIncrementId() {
  static uint64_t counter = 0;
  counter++;
  return counter;
}

template<typename K, typename C, typename E = typename C::value_type,
         typename F = std::function<K(const E&)>>
std::unique_ptr<std::unordered_map<K, std::list<E>>> XGroupBy(
    const C& container, const F& f) {
  auto collect = unique_ptr_new<std::unordered_map<K, std::list<E>>>();
  for (const E& elem : container) { (*collect)[f(elem)].push_back(elem); }
  return collect;
}

template<typename NV, typename C, typename K = typename C::key_type,
         typename V = typename C::mapped_type,
         typename F = std::function<NV(const V&)>>
std::unique_ptr<std::unordered_map<K, NV>> XAssocVMap(const C& container,
                                                      const F& f) {
  auto collect = unique_ptr_new<std::unordered_map<K, NV>>();
  for (const auto& p : container) { (*collect)[p.first] = f(p.second); }
  return collect;
}

template<typename C, typename T = typename C::const_iterator>
T XAssocKMin(const C& container) {
  auto itt = container.begin();

  if (itt != container.end()) {
    auto jtt = itt;
    for (jtt++; jtt != container.end(); jtt++) {
      if (jtt->first < itt->first) { itt = jtt; }
    }
  }

  return itt;
}

template<typename E, typename C, typename F = std::function<bool(const E&)>>
std::unique_ptr<std::list<E>> XFilter(const C& container, const F& f) {
  auto collect = unique_ptr_new<std::list<E>>();
  for (const E& elem : container) {
    if (f(elem)) { collect->push_back(elem); }
  }
  return collect;
}

template<typename K, typename C,
         typename E = std::pair<typename C::key_type, typename C::mapped_type>,
         typename F = std::function<K(const E&)>>
std::unique_ptr<std::unordered_set<K>> XAssocDistinct(const C& container,
                                                      const F& f) {
  auto collect = unique_ptr_new<std::unordered_set<K>>();
  for (const E& elem : container) { collect->insert(f(elem)); }
  return collect;
}

template<typename K, typename C, typename E = typename C::value_type,
         typename F = std::function<K(const E&)>>
std::unique_ptr<std::unordered_set<K>> XDistinct(const C& container,
                                                 const F& f) {
  auto collect = unique_ptr_new<std::unordered_set<K>>();
  for (const E& elem : container) { collect->insert(f(elem)); }
  return collect;
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

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_UTILS_UTILS_H_
