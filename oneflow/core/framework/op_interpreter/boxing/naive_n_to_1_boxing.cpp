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

Maybe<Symbol<cfg::NdSbp>> GetBroadcastNdSbp() {
  cfg::NdSbp broadcast_nd_sbp;
  broadcast_nd_sbp.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
  return SymbolOf(broadcast_nd_sbp);
}

auto* CachedGetBroadcastNdSbp = DECORATE(&GetBroadcastNdSbp, ThreadLocal);

Maybe<void> RawCheckNaiveBTo1(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(out->placement()->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBroadcastNdSbp(in->nd_sbp()));
  CHECK_OR_RETURN(in->placement()->Bigger(*out->placement()));
  CHECK_EQ_OR_RETURN(out->placement()->device_type(), DeviceType::kGPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckNaiveBTo1 = DECORATE(&RawCheckNaiveBTo1, ThreadLocal);

Maybe<void> RawCheckNaivePTo1(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(out->placement()->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsPartialSumNdSbp(in->nd_sbp()));
  CHECK_OR_RETURN(in->placement()->Bigger(*out->placement()));
  CHECK_EQ_OR_RETURN(out->placement()->device_type(), DeviceType::kGPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckNaivePTo1 = DECORATE(&RawCheckNaivePTo1, ThreadLocal);

Maybe<void> RawCheckNaiveSTo1(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(out->placement()->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsSplitNdSbp(in->nd_sbp(), 0));
  CHECK_OR_RETURN(in->placement()->Bigger(*out->placement()));
  CHECK_EQ_OR_RETURN(out->placement()->device_type(), DeviceType::kGPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckNaiveSTo1 = DECORATE(&RawCheckNaiveSTo1, ThreadLocal);

}  // namespace

Maybe<one::Tensor> NaiveBTo1(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                             Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());

  std::shared_ptr<one::Tensor> local_tensor = JUST(tensor->cur_rank_phy_tensor());
  return JUST(one::functional::LocalToConsistent(local_tensor, out->placement(),
                                                 *JUST(GetSbpList(out->nd_sbp())), *tensor->shape(),
                                                 tensor->dtype()));
}

COMMAND(RegisterBoxingFunction("naive-b-to-1", CheckNaiveBTo1, &NaiveBTo1));

Maybe<one::Tensor> NaivePTo1(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                             Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());

  Symbol<cfg::NdSbp> broadcast_nd_sbp = JUST(CachedGetBroadcastNdSbp());
  const auto& broadcast_in_placed_nd_sbp =
      JUST(PlacedNdSbp::New(broadcast_nd_sbp, in->placement()));
  const auto& NcclPToBBoxingFunction =
      *JUST(GetBoxingFunction("nccl-p-to-b", in, broadcast_in_placed_nd_sbp));
  std::shared_ptr<one::Tensor> broadcast_input =
      JUST(NcclPToBBoxingFunction(tensor, in, broadcast_in_placed_nd_sbp));

  const auto& NaiveBTo1 = *JUST(GetBoxingFunction("naive-b-to-1", broadcast_in_placed_nd_sbp, out));
  return JUST(NaiveBTo1(broadcast_input, broadcast_in_placed_nd_sbp, out));
}

COMMAND(RegisterBoxingFunction("naive-p-to-1", CheckNaivePTo1, &NaivePTo1));

Maybe<one::Tensor> NaiveSTo1(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                             Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());

  Symbol<cfg::NdSbp> broadcast_nd_sbp = JUST(CachedGetBroadcastNdSbp());
  const auto& broadcast_in_placed_nd_sbp =
      JUST(PlacedNdSbp::New(broadcast_nd_sbp, in->placement()));
  const auto& NcclSToBBoxingFunction =
      *JUST(GetBoxingFunction("nccl-s-to-b", in, broadcast_in_placed_nd_sbp));
  std::shared_ptr<one::Tensor> broadcast_input =
      JUST(NcclSToBBoxingFunction(tensor, in, broadcast_in_placed_nd_sbp));

  const auto& NaiveBTo1 = *JUST(GetBoxingFunction("naive-b-to-1", broadcast_in_placed_nd_sbp, out));
  return JUST(NaiveBTo1(broadcast_input, broadcast_in_placed_nd_sbp, out));
}

COMMAND(RegisterBoxingFunction("naive-s-to-1", CheckNaiveSTo1, &NaiveSTo1));

}  // namespace oneflow
