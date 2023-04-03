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
#include "oneflow/core/common/id_pairs.h"

#include "oneflow/core/common/hash.h"

namespace oneflow {

void InitIdPairs(const std::unordered_set<std::pair<int64_t, int64_t>>& pairs, IdPairs* proto) {
  for (const auto& pair : pairs) {
    auto* proto_pair = proto->mutable_int64_pair()->Add();
    proto_pair->set_first(pair.first);
    proto_pair->set_second(pair.second);
  }
}

void MergeIdPairs(const IdPairs& id_pairs, std::unordered_set<std::pair<int64_t, int64_t>>* pairs) {
  for (const auto& pair : id_pairs.int64_pair()) {
    pairs->emplace(std::make_pair(pair.first(), pair.second()));
  }
}

}  // namespace oneflow
