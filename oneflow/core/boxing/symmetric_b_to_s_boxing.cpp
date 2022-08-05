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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/job/nd_sbp_util.h"

namespace oneflow {

namespace {

bool RawIsBroadcastSbp(Symbol<SbpParallel> sbp_parallel) {
  return sbp_parallel->has_broadcast_parallel();
}

static constexpr auto* IsBroadcastSbp = DECORATE(&RawIsBroadcastSbp, ThreadLocalCached);

bool RawIsSplitSbp(Symbol<SbpParallel> sbp_parallel) { return sbp_parallel->has_split_parallel(); }

static constexpr auto* IsSplitSbp = DECORATE(&RawIsSplitSbp, ThreadLocalCached);

// NOLINTBEGIN(maybe-need-error-msg)
Maybe<void> RawCheckSymmetricB2S(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                 const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(IsBroadcastSbp(SymbolOf(in->nd_sbp()->sbp_parallel(0))));
  CHECK_OR_RETURN(IsSplitSbp(SymbolOf(out->nd_sbp()->sbp_parallel(0))));

  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(in->placement()->device_type() == DeviceType::kCPU
                  || in->placement()->device_type() == DeviceType::kCUDA);
  return Maybe<void>::Ok();
}
// NOLINTEND(maybe-need-error-msg)

static constexpr auto* CheckSymmetricB2S =
    DECORATE(&RawCheckSymmetricB2S, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> SymmetricB2S(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                                Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp())
      << Error::RuntimeError() << "The sbp of input tensor (" << NdSbpToString(tensor_nd_sbp)
      << ") must match the input sbp (" << NdSbpToString(in->nd_sbp()) << ")";
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement())
      << Error::RuntimeError() << "The placement of input tensor ("
      << *JUST(PlacementToString(tensor_placement)) << ") must match the input placement ("
      << *JUST(PlacementToString(in->placement())) << ")";

  const auto& local_shape = *tensor->shape();
  std::shared_ptr<one::Tensor> local_tensor = JUST(tensor->cur_rank_phy_tensor());

  const auto& parallel_id = JUST(GetParallelId4CurrentProcessCtx(tensor_placement));

  if (parallel_id->has_value()) {
    const TensorSliceView& in_slice = GetTensorSliceView4ParallelId(
        *tensor_placement->hierarchy(), *tensor_nd_sbp, local_shape, JUST(*parallel_id));
    CHECK(!in_slice.IsEmpty());
    const TensorSliceView& out_slice = GetTensorSliceView4ParallelId(
        *tensor_placement->hierarchy(), *out->nd_sbp(), local_shape, JUST(*parallel_id));
    CHECK(!out_slice.IsEmpty());
    const TensorSliceView& intersection = out_slice.Intersect(in_slice);
    CHECK(!intersection.IsEmpty());
    const std::vector<Range>& range_vec = intersection.range_vec();
    std::vector<int64_t> start;
    std::vector<int64_t> stop;
    std::vector<int64_t> step(range_vec.size(), 1);
    for (const auto& range : range_vec) {
      start.emplace_back(range.begin());
      stop.emplace_back(range.end());
    }
    local_tensor = JUST(one::functional::Slice(local_tensor, start, stop, step,
                                               /*enable_view_slice=*/false));
  }

  return JUST(one::functional::LocalToGlobal(
      local_tensor, out->placement(), *JUST(GetSbpList(out->nd_sbp())), *tensor->shape(),
      tensor->dtype(), /* sync_data */ false, /*copy=*/false));
}

COMMAND(RegisterBoxingFunction("symmetric-b-to-s", CheckSymmetricB2S, &SymmetricB2S));

}  // namespace oneflow
