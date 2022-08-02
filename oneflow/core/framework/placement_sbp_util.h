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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/common/stride.h"

namespace oneflow {

class Shape;
class Stride;
class ParallelDesc;
class PlacedNdSbp;

namespace one {

class GlobalTensorMeta;

}

// 1) src_nd_sbp.sbp_parallel_size() == 1
// 2) dst_nd_sbp.sbp_parallel_size() == 1
struct NaiveBoxingTransformation {
  Symbol<one::GlobalTensorMeta> global_tensor_meta;
  Symbol<NdSbp> dst_nd_sbp;
};

namespace private_details {

Maybe<std::vector<int64_t>> GetSelectedParallelIds(const Shape& hierarchy_shape,
                                                   const std::vector<int>& axis2is_selected,
                                                   int64_t parallel_id);

Maybe<std::tuple<std::shared_ptr<const Shape>, Symbol<NdSbp>, Symbol<NdSbp>>>
CalcDecomposableEquivalentShapeAndNdSbpPair(const Shape& shape, const Shape& hierarchy,
                                            Symbol<NdSbp> src_nd_sbp, Symbol<NdSbp> dst_nd_sbp);

Maybe<Symbol<ParallelDesc>> GetBroadcastSubParallelDesc(Symbol<ParallelDesc> parallel_desc,
                                                        Symbol<NdSbp> nd_sbp);

Maybe<std::vector<NaiveBoxingTransformation>> DecomposeIntoNaiveTransformations(
    Symbol<one::GlobalTensorMeta> tensor_meta, Symbol<NdSbp> dst_nd_sbp);

Maybe<bool> IsNdSbpBoxingAcyclic(Symbol<NdSbp> src_nd_sbp, Symbol<NdSbp> dst_nd_sbp);

Maybe<std::vector<int64_t>> GetNdSbpValidTransformationAxisSequence(Symbol<NdSbp> src_nd_sbp,
                                                                    Symbol<NdSbp> dst_nd_sbp);

Maybe<Symbol<one::GlobalTensorMeta>> CalcSubGlobalTensorMeta(
    Symbol<one::GlobalTensorMeta> tensor_meta, Symbol<ParallelDesc> sub_parallel_desc,
    Symbol<NdSbp> sub_nd_sbp);

Maybe<Symbol<ParallelDesc>> CalcSubParallelDesc4Axis(Symbol<ParallelDesc> parallel_desc, int axis);

}  // namespace private_details

extern Maybe<void> (*CheckIsNdSbpBoxingAcyclic)(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out);

extern Maybe<void> (*CheckIsNdSbpBoxingAcyclicWithDecompose)(Symbol<PlacedNdSbp> in,
                                                             Symbol<PlacedNdSbp> out,
                                                             const Shape& logical_shape);

int64_t CalcIndex4Axis(int64_t offset, const Stride& stride, int axis);

static constexpr auto* GetSubGlobalTensorMeta =
    DECORATE(&private_details::CalcSubGlobalTensorMeta, ThreadLocal);

static constexpr auto* GetBroadcastSubParallelDesc =
    DECORATE(&private_details::GetBroadcastSubParallelDesc, ThreadLocal);

static constexpr auto* DecomposeIntoNaiveTransformations =
    DECORATE(&private_details::DecomposeIntoNaiveTransformations, ThreadLocal);

static constexpr auto* CalcSubParallelDesc4Axis =
    DECORATE(&private_details::CalcSubParallelDesc4Axis, ThreadLocal);

Maybe<std::unordered_map<int64_t, Symbol<ParallelDesc>>> GetBroadcastGroup(
    Symbol<ParallelDesc> src_parallel_desc, Symbol<ParallelDesc> dst_parallel_desc);

Maybe<std::unordered_map<int64_t, Symbol<ParallelDesc>>> GetBroadcastGroupWithoutAcrossNode(
    Symbol<ParallelDesc> src_parallel_desc, Symbol<ParallelDesc> dst_parallel_desc);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_PLACEMENT_SBP_UTIL_H_
