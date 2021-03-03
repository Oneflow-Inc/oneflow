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
#ifndef ONEFLOW_CORE_COMMON_CONTAINER_UTIL_H_
#define ONEFLOW_CORE_COMMON_CONTAINER_UTIL_H_

#include <unordered_map>
#include <unordered_set>
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

#ifndef USING_HASH_CONTAINER
template<typename Key, typename T, typename Hash = std::hash<Key>>
using HashMap = std::unordered_map<Key, T, Hash>;

template<typename Key, typename Hash = std::hash<Key>>
using HashSet = std::unordered_set<Key, Hash>;
#define USING_HASH_CONTAINER
#endif  // USING_HASH_CONTAINER

template<typename MapT, typename KeyT>
scalar_or_const_ref_t<typename MapT::mapped_type> MapAtOrDefault(const MapT& map, const KeyT& key) {
  const auto& iter = map.find(key);
  if (iter == map.end()) {
    static typename MapT::mapped_type default_val;
    return default_val;
  }
  return iter->second;
}

template<typename MapT, typename KeyT>
Maybe<scalar_or_const_ref_t<typename MapT::mapped_type>> MapAt(const MapT& map, const KeyT& key) {
  const auto& iter = map.find(key);
  CHECK_OR_RETURN(iter != map.end());
  return iter->second;
}

template<typename MapT, typename KeyT>
Maybe<typename MapT::mapped_type*> MapAt(MapT* map, const KeyT& key) {
  const auto& iter = map->find(key);
  CHECK_OR_RETURN(iter != map->end());
  return &iter->second;
}

template<typename VecT>
Maybe<scalar_or_const_ref_t<typename VecT::value_type>> VectorAt(const VecT& vec, int64_t index) {
  CHECK_GE_OR_RETURN(index, 0);
  CHECK_LT_OR_RETURN(index, vec.size());
  return vec.at(index);
}

template<typename VecT>
Maybe<typename VecT::value_type*> VectorAt(VecT* vec, int64_t index) {
  CHECK_GE_OR_RETURN(index, 0);
  CHECK_LT_OR_RETURN(index, vec->size());
  return &vec->at(index);
}

template<typename T>
std::string Join(const T& vec, const std::string& glue) {
  std::string str;
  bool not_first = false;
  for (const auto& elem : vec) {
    if (not_first) { str += glue; }
    str += elem;
    not_first = true;
  }
  return str;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CONTAINER_UTIL_H_
