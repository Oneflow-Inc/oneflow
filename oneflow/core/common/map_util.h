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
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

template<typename MapT, typename KeyT>
Maybe<scalar_or_const_ref_t<typename MapT::mapped_type>> MapAt(const MapT& map, const KeyT& key) {
  const auto& iter = map.find(key);
  CHECK_OR_RETURN(iter != map.end());
  return iter->second;
}

}  // namespace oneflow
