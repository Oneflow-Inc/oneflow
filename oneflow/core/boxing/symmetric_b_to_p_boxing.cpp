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
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {

namespace {

Maybe<void> RawCheckSymmetricBToP(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                  const Shape& logical_shape) {
  // NOLINTBEGIN(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(NdSbpIsAllBroadcast(*in->nd_sbp()));
  CHECK_OR_RETURN(NdSbpIsAllPartialSum(*out->nd_sbp()));

  CHECK_OR_RETURN(in->placement() == out->placement());
  // NOLINTEND(maybe-need-error-msg)
  return Maybe<void>::Ok();
}

static constexpr auto* CheckSymmetricBToP =
    DECORATE(&RawCheckSymmetricBToP, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> SymmetricBToP(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
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
  if (root == GlobalProcessCtx::Rank()) {
    // do nothing
  } else {
    local_tensor = JUST(one::functional::ZerosLike(local_tensor));
  }
  return JUST(one::functional::LocalToGlobal(
      local_tensor, out->placement(), *JUST(GetSbpList(out->nd_sbp())), *tensor->shape(),
      tensor->dtype(), /* sync_data */ false, /*copy=*/true));
}

COMMAND(RegisterBoxingFunction("symmetric-b-to-p", CheckSymmetricBToP, &SymmetricBToP));

}  // namespace oneflow
