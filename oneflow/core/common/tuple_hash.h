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
#include "oneflow/core/common/cplusplus_14.h"
#include "oneflow/core/common/util.h"

namespace std {

template<typename... T>
struct hash<std::tuple<T...>> final {
  size_t operator()(const std::tuple<T...>& val) const {
    size_t seed = 0;

    do_hash(val, seed, std::index_sequence_for<T...>{});

    return seed;
  }

 private:
  template<size_t... I>
  void do_hash(const std::tuple<T...>& val, size_t& seed, std::index_sequence<I...>) const {
    __attribute__((__unused__)) int dummy[] = {
        (oneflow::HashCombineTyped<T>(&seed, std::get<I>(val)), 0)...};
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_TUPLE_HASH_H_
