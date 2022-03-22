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
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {

namespace {

bool IsAllBroadcastNdSbp(Symbol<NdSbp> nd_sbp) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!sbp_parallel.has_broadcast_parallel()) { return false; }
  }
  return true;
}

bool IsAllPartialSumNdSbp(Symbol<NdSbp> nd_sbp) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!sbp_parallel.has_partial_sum_parallel()) { return false; }
  }
  return true;
}

Maybe<void> RawCheckSymmetricBToP(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                  const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(IsAllBroadcastNdSbp(in->nd_sbp()));
  CHECK_OR_RETURN(IsAllPartialSumNdSbp(out->nd_sbp()));

  CHECK_OR_RETURN(in->placement() == out->placement());
  return Maybe<void>::Ok();
}

static constexpr auto* CheckSymmetricBToP =
    DECORATE(&RawCheckSymmetricBToP, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> SymmetricBToP(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                                 Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());

  int64_t root = JUST(tensor_placement->MachineId4ParallelId(0));
  std::shared_ptr<one::Tensor> local_tensor = JUST(tensor->cur_rank_phy_tensor());
  if (root == GlobalProcessCtx::Rank()) {
    // do nothing
  } else {
    const std::string& device_type = Device::Type4DeviceTag(tensor_placement->device_tag());
    local_tensor = JUST(one::functional::ZerosLike(local_tensor));
  }
  return JUST(one::functional::LocalToConsistent(local_tensor, out->placement(),
                                                 *JUST(GetSbpList(out->nd_sbp())), *tensor->shape(),
                                                 tensor->dtype()));
}

COMMAND(RegisterBoxingFunction("symmetric-b-to-p", CheckSymmetricBToP, &SymmetricBToP));

}  // namespace oneflow
