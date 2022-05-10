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

Maybe<void> RawCheckCclSToB(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                            const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(IsSplitSbp(in->nd_sbp()->sbp_parallel(0)));
  CHECK_OR_RETURN(IsBroadcastSbp(out->nd_sbp()->sbp_parallel(0)));
  CHECK_EQ_OR_RETURN(in->placement()->device_tag(), out->placement()->device_tag());
  return Maybe<void>::Ok();
}

static constexpr auto* CheckCclSToB = DECORATE(&RawCheckCclSToB, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> CclSToB(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                           Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  std::shared_ptr<one::Tensor> processed_in_tensor = JUST(PreprocessInputTensor4SliceBoxing(
      tensor, /* log_prefix */ "\t\tInternal boxing of naive-s-to-b, "));

  Symbol<ParallelDesc> new_out_placement = JUST(ReplaceDeviceType(
      out->placement(), JUST(processed_in_tensor->parallel_desc())->device_type()));

  std::shared_ptr<one::Tensor> local_tensor = JUST(processed_in_tensor->cur_rank_phy_tensor());
  {
    const auto& in_parallel_id =
        JUST(GetParallelId4CurrentProcessCtx(JUST(processed_in_tensor->parallel_desc())));
    const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(new_out_placement));
    if (in_parallel_id->has_value() || out_parallel_id->has_value()) {
      local_tensor = JUST(one::functional::EagerSToB(
          local_tensor, JUST(processed_in_tensor->parallel_desc()), new_out_placement,
          *JUST(GetSbpList(tensor_nd_sbp)), *tensor->shape()));
    }
  }

  const auto& sbp_list = JUST(GetSbpList(out->nd_sbp()));
  std::shared_ptr<one::Tensor> out_tensor = JUST(one::functional::LocalToConsistent(
      local_tensor, new_out_placement, *sbp_list, *tensor->shape(), tensor->dtype()));

  return JUST(PostprocessOutputTensor4SliceBoxing(
      out_tensor, out, /* log_prefix */ "\t\tInternal boxing of naive-b-to-s, "));
}

COMMAND(RegisterBoxingFunction("naive-s-to-b", CheckCclSToB, &CclSToB));

}  // namespace oneflow
