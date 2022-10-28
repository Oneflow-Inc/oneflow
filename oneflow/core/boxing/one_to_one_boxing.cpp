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
#include "oneflow/user/kernels/communicate_util.h"

namespace oneflow {

namespace {

// NOLINTBEGIN(maybe-need-error-msg)
Maybe<void> RawCheckNaiveOneToOne(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                  const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->placement()->parallel_num(), 1);
  CHECK_EQ_OR_RETURN(out->placement()->parallel_num(), 1);
  CHECK_EQ_OR_RETURN(in->placement()->device_tag(), out->placement()->device_tag());
  CHECK_OR_RETURN(in->placement() != out->placement());
  CHECK_OR_RETURN(IsSendAndRecvRegistered(in->placement()->device_type()));  // NOLINT
  return Maybe<void>::Ok();
}
// NOLINTEND(maybe-need-error-msg)

static constexpr auto* CheckNaiveOneToOne =
    DECORATE(&RawCheckNaiveOneToOne, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> NaiveOneToOne(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
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

  std::shared_ptr<one::Tensor> local_tensor = JUST(tensor->cur_rank_phy_tensor());
  int64_t src = JUST(tensor_placement->MachineId4ParallelId(0));
  int64_t dst = JUST(out->placement()->MachineId4ParallelId(0));

  bool copy = true;
  if (src != dst) {
    copy = false;
    if (GlobalProcessCtx::Rank() == src) {
      JUST(one::functional::Send(local_tensor, dst, /* send_meta */ false));
    }
    if (GlobalProcessCtx::Rank() == dst) {
      local_tensor = JUST(one::functional::Recv(src, *tensor->shape(), tensor->dtype(),
                                                JUST(local_tensor->device()), NullOpt));
    }
  }
  return JUST(one::functional::LocalToGlobal(
      local_tensor, out->placement(), *JUST(GetSbpList(out->nd_sbp())), *tensor->shape(),
      tensor->dtype(), /* sync_data */ false, /*copy=*/copy));
}

COMMAND(RegisterBoxingFunction("naive-1-to-1", CheckNaiveOneToOne, &NaiveOneToOne));

}  // namespace oneflow
