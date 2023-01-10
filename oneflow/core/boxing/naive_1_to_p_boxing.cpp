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

namespace oneflow {

namespace {

bool NdSbpIsAllPartialSum(Symbol<NdSbp> nd_sbp) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!sbp_parallel.has_partial_sum_parallel()) { return false; }
  }
  return true;
}

// NOLINTBEGIN(maybe-need-error-msg)
Maybe<void> RawCheckNaive1ToP(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                              const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->placement()->parallel_num(), 1);
  CHECK_OR_RETURN(NdSbpIsAllPartialSum(out->nd_sbp()));
  CHECK_OR_RETURN(out->placement()->Bigger(*in->placement()));
  return Maybe<void>::Ok();
}
// NOLINTEND(maybe-need-error-msg)

static constexpr auto* CheckNaive1ToP = DECORATE(&RawCheckNaive1ToP, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> Naive1ToP(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
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

  int64_t root = JUST(tensor_placement->MachineId4ParallelId(0));
  std::shared_ptr<one::Tensor> local_tensor = JUST(tensor->cur_rank_phy_tensor());
  const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out->placement()));
  if (root == GlobalProcessCtx::Rank() || !out_parallel_id->has_value()) {
    // do nothing
  } else {
    const std::string& device_type = tensor_placement->device_tag();
    local_tensor = JUST(one::functional::Constant(*tensor->shape(), 0, tensor->dtype(),
                                                  JUST(Device::New(device_type))));
  }
  return JUST(one::functional::LocalToGlobal(
      local_tensor, out->placement(), *JUST(GetSbpList(out->nd_sbp())), *tensor->shape(),
      tensor->dtype(), /* sync_data */ false, /*copy=*/true));
}

COMMAND(RegisterBoxingFunction("naive-1-to-p", CheckNaive1ToP, &Naive1ToP));

}  // namespace oneflow
