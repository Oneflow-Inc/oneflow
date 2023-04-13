/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMMON_HASH_H_
#define ONEFLOW_CORE_COMMON_HASH_H_
#include <functional>
#include <complex>

namespace oneflow {

inline size_t HashCombine(size_t lhs, size_t rhs) {
  return lhs ^ (rhs + 0x9e3779b9 + (lhs << 6U) + (lhs >> 2U));
}

inline void HashCombine(size_t* seed, size_t hash) { *seed = HashCombine(*seed, hash); }

template<typename... T>
inline void AddHash(size_t* seed, const T&... v) {
  __attribute__((__unused__)) int dummy[] = {(HashCombine(seed, std::hash<T>()(v)), 0)...};
}

template<typename T, typename... Ts>
inline size_t Hash(const T& v1, const Ts&... vn) {
  size_t seed = std::hash<T>()(v1);

  AddHash<Ts...>(&seed, vn...);

  return seed;
}

}  // namespace oneflow

namespace std {

template<typename T0, typename T1>
struct hash<std::pair<T0, T1>> {
  std::size_t operator()(const std::pair<T0, T1>& p) const {
    return oneflow::Hash<T0, T1>(p.first, p.second);
  }
};

template<typename T>
struct hash<std::vector<T>> {
  std::size_t operator()(const std::vector<T>& vec) const {
    std::size_t hash_value = vec.size();
    for (const auto& elem : vec) { oneflow::AddHash<T>(&hash_value, elem); }
    return hash_value;
  }
};

template<typename T>
struct hash<std::complex<T>> {
  size_t operator()(const std::complex<T>& c) const { return oneflow::Hash(c.real(), c.imag()); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_HASH_H_
