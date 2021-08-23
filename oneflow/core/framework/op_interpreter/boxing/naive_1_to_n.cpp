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
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {

namespace {

Maybe<Symbol<cfg::NdSbp>> GetPartialSumNdSbp() {
  cfg::NdSbp partial_sum_nd_sbp;
  partial_sum_nd_sbp.mutable_sbp_parallel()->Add()->mutable_partial_sum_parallel();
  return SymbolOf(partial_sum_nd_sbp);
}

auto* CachedGetPartialSumNdSbp = DECORATE(&GetPartialSumNdSbp, ThreadLocal);

Maybe<void> RawCheckNaive1ToP(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(in->placement()->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsPartialSumNdSbp(out->nd_sbp()));
  CHECK_OR_RETURN(out->placement()->Bigger(*in->placement()));
  return Maybe<void>::Ok();
}

static constexpr auto* CheckNaive1ToP = DECORATE(&RawCheckNaive1ToP, ThreadLocal);

Maybe<void> RawCheckNaive1ToB(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(in->placement()->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBroadcastNdSbp(out->nd_sbp()));
  CHECK_OR_RETURN(out->placement()->Bigger(*in->placement()));
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), DeviceType::kGPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckNaive1ToB = DECORATE(&RawCheckNaive1ToB, ThreadLocal);

Maybe<void> RawCheckNaive1ToS(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(in->placement()->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsSplitNdSbp(out->nd_sbp(), 0));
  CHECK_OR_RETURN(out->placement()->Bigger(*in->placement()));
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), DeviceType::kGPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckNaive1ToS = DECORATE(&RawCheckNaive1ToS, ThreadLocal);

}  // namespace

Maybe<one::Tensor> Naive1ToP(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                             Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());

  int64_t root = JUST(tensor_placement->MachineId4ParallelId(0));
  std::shared_ptr<one::Tensor> local_tensor = JUST(tensor->cur_rank_phy_tensor());
  const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out->placement()));
  if (root == GlobalProcessCtx::Rank() || !out_parallel_id->has_value()) {
    // do nothing
  } else {
    std::string device_type = Device::Type4DeviceTag(tensor_placement->device_tag());
    local_tensor = JUST(one::functional::Constant(*tensor->shape(), 0, tensor->dtype(),
                                                  JUST(Device::New(device_type))));
  }
  return JUST(one::functional::LocalToConsistent(local_tensor, out->placement(),
                                                 *JUST(GetSbpList(out->nd_sbp())), *tensor->shape(),
                                                 tensor->dtype()));
}

COMMAND(RegisterBoxingFunction("naive-1-to-p", CheckNaive1ToP, &Naive1ToP));

Maybe<one::Tensor> Naive1ToB(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                             Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());

  Symbol<cfg::NdSbp> partial_sum_nd_sbp = JUST(CachedGetPartialSumNdSbp());
  const auto& partial_sum_out_placed_nd_sbp =
      JUST(PlacedNdSbp::New(partial_sum_nd_sbp, out->placement()));
  const auto& Naive1ToPBoxingFunction =
      *JUST(GetBoxingFunction("naive-1-to-p", in, partial_sum_out_placed_nd_sbp));
  std::shared_ptr<one::Tensor> partial_sum_output =
      JUST(Naive1ToPBoxingFunction(tensor, in, partial_sum_out_placed_nd_sbp));

  const auto& NcclPToBBoxingFunction =
      *JUST(GetBoxingFunction("nccl-p-to-b", partial_sum_out_placed_nd_sbp, out));

  return JUST(NcclPToBBoxingFunction(partial_sum_output, partial_sum_out_placed_nd_sbp, out));
}

COMMAND(RegisterBoxingFunction("naive-1-to-b", CheckNaive1ToB, &Naive1ToB));

Maybe<one::Tensor> Naive1ToS(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                             Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());

  Symbol<cfg::NdSbp> partial_sum_nd_sbp = JUST(CachedGetPartialSumNdSbp());
  const auto& partial_sum_out_placed_nd_sbp =
      JUST(PlacedNdSbp::New(partial_sum_nd_sbp, out->placement()));
  const auto& Naive1ToPBoxingFunction =
      *JUST(GetBoxingFunction("naive-1-to-p", in, partial_sum_out_placed_nd_sbp));
  std::shared_ptr<one::Tensor> partial_sum_output =
      JUST(Naive1ToPBoxingFunction(tensor, in, partial_sum_out_placed_nd_sbp));

  const auto& NcclPToSBoxingFunction =
      *JUST(GetBoxingFunction("nccl-p-to-s", partial_sum_out_placed_nd_sbp, out));

  return JUST(NcclPToSBoxingFunction(partial_sum_output, partial_sum_out_placed_nd_sbp, out));
}

COMMAND(RegisterBoxingFunction("naive-1-to-s", CheckNaive1ToS, &Naive1ToS));

}  // namespace oneflow
