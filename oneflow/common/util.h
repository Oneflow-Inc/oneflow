#ifndef _COMMON_UTIL_H_
#define _COMMON_UTIL_H_
#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>

//#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
//  TypeName(const TypeName&) = delete; \
//  TypeName& operator=(const TypeName&) = delete

namespace caffe {

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& s) {
  os << "{ ";
  for (const T& t: s) {
    os << t << " ";
  }
  return os << "}";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_set<T>& s) {
  os << "{ ";
  for (const T& t: s) {
    os << t << " ";
  }
  return os << "}";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[ ";
  for (const T& t: v) {
    os << t << " ";
  }
  return os << "]";
}

template<typename U, typename T, typename Fn>
std::vector<U> Map(const std::vector<T>& original, Fn fn) {
  std::vector<U> res;
  res.resize(original.size());
  std::transform(original.begin(), original.end(), res.begin(), fn);
  return res;
}

template<typename Iterable, typename Fn>
void Iter(const Iterable& original, Fn fn) {
  std::for_each(original.begin(), original.end(), fn);
}

}  // namespace minerva
#endif  // _COMMON_UTIL_H_

