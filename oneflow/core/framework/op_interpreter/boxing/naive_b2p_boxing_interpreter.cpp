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
#include "oneflow/core/framework/op_interpreter/boxing/naive_b2p_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {

Maybe<one::Tensor> NaiveB2PBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(in_parallel_desc == out_parallel_desc);
  int64_t root = JUST(in_parallel_desc->MachineId4ParallelId(0));
  std::shared_ptr<one::Tensor> tensor = JUST(input->cur_rank_phy_tensor());
  if (root == GlobalProcessCtx::Rank()) {
    // do nothing
  } else {
    tensor = JUST(one::functional::ZerosLike(tensor));
  }
  return one::functional::ToConsistent(tensor, out_parallel_desc, *JUST(GetSbpList(out_nd_sbp)),
                                       GetNoneSbpList());
}

namespace {

Maybe<void> RawCheckNaiveBToP(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsAllBroadcastNdSbp(in->nd_sbp()));
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsAllPartialSumNdSbp(out->nd_sbp()));

  CHECK_OR_RETURN(in->placement() == out->placement());
  return Maybe<void>::Ok();
}

static constexpr auto* CheckNaiveBToP = DECORATE(&RawCheckNaiveBToP, ThreadLocal);

}  // namespace

Maybe<one::Tensor> NaiveBToP(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
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

COMMAND(RegisterBoxingFunction("naive-b-to-p", CheckNaiveBToP, &NaiveBToP));

}  // namespace oneflow
