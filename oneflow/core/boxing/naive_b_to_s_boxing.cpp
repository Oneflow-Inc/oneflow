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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/boxing/slice_boxing_util.h"

namespace oneflow {

namespace {

bool RawIsSplitSbp(Symbol<SbpParallel> sbp_parallel) { return sbp_parallel->has_split_parallel(); }

static constexpr auto* IsSplitSbp = DECORATE(&RawIsSplitSbp, ThreadLocalCached);

bool RawIsBroadcastSbp(Symbol<SbpParallel> sbp_parallel) {
  return sbp_parallel->has_broadcast_parallel();
}

static constexpr auto* IsBroadcastSbp = DECORATE(&RawIsBroadcastSbp, ThreadLocalCached);

// NOLINTBEGIN(maybe-need-error-msg)
Maybe<void> RawCheckNaiveBToS(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                              const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);

  CHECK_OR_RETURN(IsBroadcastSbp(in->nd_sbp()->sbp_parallel(0)));
  CHECK_OR_RETURN(IsSplitSbp(out->nd_sbp()->sbp_parallel(0)));

  CHECK_EQ_OR_RETURN(in->placement()->device_tag(), out->placement()->device_tag());
  return Maybe<void>::Ok();
}
// NOLINTEND(maybe-need-error-msg)

static constexpr auto* CheckNaiveBToS = DECORATE(&RawCheckNaiveBToS, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> NaiveBToS(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
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
  const auto& sbp_list = JUST(GetSbpList(out->nd_sbp()));
  std::shared_ptr<one::Tensor> local_tensor = JUST(tensor->cur_rank_phy_tensor());
  {
    const auto& in_parallel_id = JUST(GetParallelId4CurrentProcessCtx(tensor_placement));
    const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out->placement()));
    if (in_parallel_id->has_value() || out_parallel_id->has_value()) {
      local_tensor = JUST(one::functional::EagerBToS(
          local_tensor, tensor_placement, out->placement(), *sbp_list, *tensor->shape()));
    }
  }

  return JUST(one::functional::LocalToGlobal(local_tensor, out->placement(), *sbp_list,
                                             *tensor->shape(), tensor->dtype(),
                                             /* sync_data */ false, /*copy=*/false));
}

static constexpr auto* NaiveBToSWithAutoConvert =
    EAGER_SLICE_BOXING_WARPPER(&NaiveBToS, EagerSliceBoxingType::kNaiveBToS);

COMMAND(RegisterBoxingFunction("naive-b-to-s", CheckNaiveBToS, NaiveBToSWithAutoConvert));

}  // namespace oneflow
