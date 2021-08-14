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
#ifndef ONEFLOW_CORE_FRAMEWORK_PLACEMENT_SBP_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_PLACEMENT_SBP_UTIL_H_

#include <unordered_map>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class Shape;
class ParallelDesc;

namespace cfg {

class NdSbp;

}

Maybe<Symbol<ParallelDesc>> GetBroadcastSubParallelDesc(Symbol<ParallelDesc> parallel_desc,
                                                        Symbol<cfg::NdSbp> nd_sbp);

Maybe<std::vector<int64_t>> GetBroadcastParallelIds(const Shape& hierarchy_shape,
                                                    const std::vector<bool>& dim2is_broadcast,
                                                    int64_t parallel_id);

Maybe<std::unordered_map<int64_t, Symbol<ParallelDesc>>> GetBroadcastGroup(
    Symbol<ParallelDesc> src_parallel_desc, Symbol<ParallelDesc> dst_parallel_desc);

Maybe<std::unordered_map<int64_t, Symbol<ParallelDesc>>> GetBroadcastGroupWithoutAcrossNode(
    Symbol<ParallelDesc> src_parallel_desc, Symbol<ParallelDesc> dst_parallel_desc);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_PLACEMENT_SBP_UTIL_H_
