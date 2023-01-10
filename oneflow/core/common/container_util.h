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

#include <vector>
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

template<typename MapT, typename KeyT, typename U>
scalar_or_const_ref_t<typename MapT::mapped_type> MapAt(const MapT& map, const KeyT& key,
                                                        const U& default_val) {
  const auto& iter = map.find(key);
  if (iter == map.end()) { return default_val; }
  return iter->second;
}

template<typename MapT, typename KeyT>
Maybe<scalar_or_const_ref_t<typename MapT::mapped_type>> MapAt(const MapT& map, const KeyT& key) {
  const auto& iter = map.find(key);
  CHECK_OR_RETURN(iter != map.end());
  return iter->second;
}

template<typename MapT, typename KeyT>
Maybe<typename MapT::mapped_type&> MapAt(MapT& map, const KeyT& key) {
  const auto& iter = map.find(key);
  CHECK_OR_RETURN(iter != map.end());
  return iter->second;
}

template<typename VecT>
Maybe<scalar_or_const_ref_t<typename VecT::value_type>> VectorAt(const VecT& vec,
                                                                 typename VecT::size_type index) {
  CHECK_LT_OR_RETURN(index, vec.size());
  return vec[index];
}

template<typename VecT>
Maybe<typename VecT::value_type&> VectorAt(VecT& vec, typename VecT::size_type index) {
  static_assert(!std::is_same<typename VecT::value_type, bool>::value,
                "VectorAt(vector<bool>&, size_t) is not supported.");
  CHECK_LT_OR_RETURN(index, vec.size());
  return vec[index];
}

template<>
inline Maybe<bool> VectorAt(const std::vector<bool>& vec,
                            typename std::vector<bool>::size_type index) {
  CHECK_LT_OR_RETURN(index, vec.size());
  // convert vector bool proxy to bool
  return static_cast<bool>(vec[index]);
}

template<typename T>
std::string Join(const T& con, const std::string& delimiter) {
  std::ostringstream os;
  auto b = begin(con), e = end(con);

  if (b != e) {
    std::copy(b, prev(e), std::ostream_iterator<typename T::value_type>(os, delimiter));
    b = prev(e);
  }
  if (b != e) { os << *b; }

  return os.str();
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CONTAINER_UTIL_H_
