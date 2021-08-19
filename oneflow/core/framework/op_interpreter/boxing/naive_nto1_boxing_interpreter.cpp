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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_nto1_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/collective_boxing_interpreter.h"

namespace oneflow {

namespace {

Maybe<one::UserOpExpr> EagerNcclReduce(Symbol<ParallelDesc> parallel_desc, int64_t root) {
  return one::OpBuilder("eager_nccl_reduce", *JUST(UniqueStr("eager_nccl_reduce")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Attr<int64_t>("root", root)
      .Build();
}

auto* CachedEagerNcclReduce = DECORATE(&EagerNcclReduce, ThreadLocal);

Maybe<Symbol<cfg::NdSbp>> GetBroadcastSumNdSbp() {
  cfg::NdSbp broadcast_nd_sbp;
  broadcast_nd_sbp.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
  return SymbolOf(broadcast_nd_sbp);
}

auto* CachedGetBroadcastSumNdSbp = DECORATE(&GetBroadcastSumNdSbp, ThreadLocal);

}  // namespace

Maybe<one::Tensor> NcclBTo1BoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_EQ_OR_RETURN(out_parallel_desc->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBroadcastNdSbp(in_nd_sbp));
  CHECK_OR_RETURN(in_parallel_desc->Bigger(*out_parallel_desc));
  std::shared_ptr<one::Tensor> local_tensor = JUST(input->cur_rank_phy_tensor());
  const auto& output_tensor = JUST(one::functional::ToConsistent(
      local_tensor, out_parallel_desc, *JUST(GetSbpList(out_nd_sbp)), GetNoneSbpList()));
  CHECK_OR_RETURN(output_tensor->is_consistent());
  const auto& tensor_placement = JUST(output_tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == out_parallel_desc);
  return output_tensor;
}

Maybe<one::Tensor> NcclPTo1BoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_EQ_OR_RETURN(out_parallel_desc->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsPartialSumNdSbp(in_nd_sbp));
  CHECK_OR_RETURN(in_parallel_desc->Bigger(*out_parallel_desc));
  std::shared_ptr<one::Tensor> local_tensor = JUST(input->cur_rank_phy_tensor());
  const auto& in_parallel_id = JUST(GetParallelId4CurrentProcessCtx(in_parallel_desc));
  if (in_parallel_id->has_value()) {
    int64_t root = JUST(out_parallel_desc->MachineId4ParallelId(0));
    const auto& op_expr = JUST(CachedEagerNcclReduce(in_parallel_desc, root));
    local_tensor = JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {local_tensor}));
  }
  const auto& output_tensor = JUST(one::functional::ToConsistent(
      local_tensor, out_parallel_desc, *JUST(GetSbpList(out_nd_sbp)), GetNoneSbpList()));
  CHECK_OR_RETURN(output_tensor->is_consistent());
  const auto& tensor_placement = JUST(output_tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == out_parallel_desc);
  return output_tensor;
}

Maybe<one::Tensor> NcclSTo1BoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_EQ_OR_RETURN(out_parallel_desc->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsSplitNdSbp(in_nd_sbp, 0));
  CHECK_OR_RETURN(in_parallel_desc->Bigger(*out_parallel_desc));
  Symbol<cfg::NdSbp> broadcast_nd_sbp = JUST(CachedGetBroadcastSumNdSbp());
  static std::shared_ptr<EagerBoxingInterpreter> nccl_all_reduce_boxing_interpreter =
      std::make_shared<NcclCollectiveAllGatherBoxingInterpreter>();
  static std::shared_ptr<EagerBoxingInterpreter> nccl_bto1_boxing_interpreter =
      std::make_shared<NcclBTo1BoxingInterpreter>();
  const auto& broadcast_input = JUST(nccl_all_reduce_boxing_interpreter->Interpret(
      input, in_nd_sbp, broadcast_nd_sbp, in_parallel_desc, in_parallel_desc));
  return JUST(nccl_bto1_boxing_interpreter->Interpret(broadcast_input, broadcast_nd_sbp, out_nd_sbp,
                                                      in_parallel_desc, out_parallel_desc));
}
}  // namespace oneflow
