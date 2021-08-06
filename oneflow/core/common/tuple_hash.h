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
#ifndef ONEFLOW_CORE_COMMON_TUPLE_HASH_H_
#define ONEFLOW_CORE_COMMON_TUPLE_HASH_H_

#include <tuple>
#include <functional>

namespace std {

template<typename T>
struct hash<std::tuple<T>> final {
  size_t operator()(const std::tuple<T>& val) const { return std::hash<T>()(std::get<0>(val)); }
};

template<typename T0, typename T1>
struct hash<std::tuple<T0, T1>> final {
  size_t operator()(const std::tuple<T0, T1>& val) const {
    return std::hash<T0>()(std::get<0>(val)) ^ std::hash<T1>()(std::get<1>(val));
  }
};

template<typename T0, typename T1, typename T2>
struct hash<std::tuple<T0, T1, T2>> final {
  size_t operator()(const std::tuple<T0, T1, T2>& val) const {
    return std::hash<T0>()(std::get<0>(val)) ^ std::hash<T1>()(std::get<1>(val))
           ^ std::hash<T2>()(std::get<2>(val));
  }
};

template<typename T0, typename T1, typename T2, typename T3>
struct hash<std::tuple<T0, T1, T2, T3>> final {
  size_t operator()(const std::tuple<T0, T1, T2, T3>& val) const {
    return std::hash<T0>()(std::get<0>(val)) ^ std::hash<T1>()(std::get<1>(val))
           ^ std::hash<T2>()(std::get<2>(val)) ^ std::hash<T3>()(std::get<3>(val));
  }
};

template<typename T0, typename T1, typename T2, typename T3, typename T4>
struct hash<std::tuple<T0, T1, T2, T3, T4>> final {
  size_t operator()(const std::tuple<T0, T1, T2, T3, T4>& val) const {
    return std::hash<T0>()(std::get<0>(val)) ^ std::hash<T1>()(std::get<1>(val))
           ^ std::hash<T2>()(std::get<2>(val)) ^ std::hash<T3>()(std::get<3>(val))
           ^ std::hash<T4>()(std::get<4>(val));
  }
};

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
struct hash<std::tuple<T0, T1, T2, T3, T4, T5>> final {
  size_t operator()(const std::tuple<T0, T1, T2, T3, T4, T5>& val) const {
    return std::hash<T0>()(std::get<0>(val)) ^ std::hash<T1>()(std::get<1>(val))
           ^ std::hash<T2>()(std::get<2>(val)) ^ std::hash<T3>()(std::get<3>(val))
           ^ std::hash<T4>()(std::get<4>(val)) ^ std::hash<T5>()(std::get<5>(val));
  }
};

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
struct hash<std::tuple<T0, T1, T2, T3, T4, T5, T6>> final {
  size_t operator()(const std::tuple<T0, T1, T2, T3, T4, T5, T6>& val) const {
    return std::hash<T0>()(std::get<0>(val)) ^ std::hash<T1>()(std::get<1>(val))
           ^ std::hash<T2>()(std::get<2>(val)) ^ std::hash<T3>()(std::get<3>(val))
           ^ std::hash<T4>()(std::get<4>(val)) ^ std::hash<T5>()(std::get<5>(val))
           ^ std::hash<T6>()(std::get<6>(val));
  }
};

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
         typename T7>
struct hash<std::tuple<T0, T1, T2, T3, T4, T5, T6, T7>> final {
  size_t operator()(const std::tuple<T0, T1, T2, T3, T4, T5, T6, T7>& val) const {
    return std::hash<T0>()(std::get<0>(val)) ^ std::hash<T1>()(std::get<1>(val))
           ^ std::hash<T2>()(std::get<2>(val)) ^ std::hash<T3>()(std::get<3>(val))
           ^ std::hash<T4>()(std::get<4>(val)) ^ std::hash<T5>()(std::get<5>(val))
           ^ std::hash<T6>()(std::get<6>(val)) ^ std::hash<T7>()(std::get<7>(val));
  }
};

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
         typename T7, typename T8>
struct hash<std::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8>> final {
  size_t operator()(const std::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8>& val) const {
    return std::hash<T0>()(std::get<0>(val)) ^ std::hash<T1>()(std::get<1>(val))
           ^ std::hash<T2>()(std::get<2>(val)) ^ std::hash<T3>()(std::get<3>(val))
           ^ std::hash<T4>()(std::get<4>(val)) ^ std::hash<T5>()(std::get<5>(val))
           ^ std::hash<T6>()(std::get<6>(val)) ^ std::hash<T7>()(std::get<7>(val))
           ^ std::hash<T8>()(std::get<8>(val));
  }
};

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
         typename T7, typename T8, typename T9>
struct hash<std::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>> final {
  size_t operator()(const std::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& val) const {
    return std::hash<T0>()(std::get<0>(val)) ^ std::hash<T1>()(std::get<1>(val))
           ^ std::hash<T2>()(std::get<2>(val)) ^ std::hash<T3>()(std::get<3>(val))
           ^ std::hash<T4>()(std::get<4>(val)) ^ std::hash<T5>()(std::get<5>(val))
           ^ std::hash<T6>()(std::get<6>(val)) ^ std::hash<T7>()(std::get<7>(val))
           ^ std::hash<T8>()(std::get<8>(val)) ^ std::hash<T9>()(std::get<9>(val));
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_TUPLE_HASH_H_
