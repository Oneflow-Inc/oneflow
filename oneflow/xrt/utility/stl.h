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
#ifndef ONEFLOW_XRT_UTILITY_STL_H_
#define ONEFLOW_XRT_UTILITY_STL_H_

#include <list>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "google/protobuf/map.h"

namespace oneflow {
namespace xrt {
namespace util {

template<typename T>
using Vector = std::vector<T>;

template<typename T>
using List = std::list<T>;

template<typename K, typename T, typename Hash = std::hash<K>, typename Comp = std::equal_to<K>>
using Map = std::unordered_map<K, T, Hash, Comp>;

template<typename K, typename Hash = std::hash<K>, typename Comp = std::equal_to<K>>
using Set = std::unordered_set<K, Hash, Comp>;

template<typename T>
using Stack = std::stack<T>;

template<typename T>
using Queue = std::queue<T>;

template<typename K, typename T>
using PbMap = google::protobuf::Map<K, T>;

template<typename K, typename T>
inline PbMap<K, T> ConvertToPbMap(const Map<K, T>& stdmap) {
  PbMap<K, T> pbmap;
  for (const auto& it : stdmap) { pbmap[it.first] = it.second; }
  return std::move(pbmap);
}

template<typename T>
using PbVector = google::protobuf::RepeatedPtrField<T>;

}  // namespace util
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_UTILITY_STL_H_
