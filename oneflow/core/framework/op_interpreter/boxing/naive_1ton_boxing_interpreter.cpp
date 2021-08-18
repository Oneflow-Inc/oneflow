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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_1ton_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"

namespace oneflow {

namespace {

Maybe<Symbol<cfg::NdSbp>> GetPartialSumNdSbp() {
  cfg::NdSbp broadcast_nd_sbp;
  broadcast_nd_sbp.mutable_sbp_parallel()->Add()->mutable_partial_sum_parallel();
  return SymbolOf(broadcast_nd_sbp);
}

}  // namespace

Maybe<one::Tensor> Nccl1ToBBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_EQ_OR_RETURN(in_parallel_desc->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBroadcastNdSbp(out_nd_sbp));
  Symbol<cfg::NdSbp> partial_sum_nd_sbp = JUST(GetPartialSumNdSbp());
  static std::shared_ptr<EagerBoxingInterpreter> nccl_1top_boxing_interpreter =
      std::make_shared<Nccl1ToPBoxingInterpreter>();
  const auto& partial_sum_input = JUST(nccl_1top_boxing_interpreter->Interpret(
      input, in_nd_sbp, partial_sum_nd_sbp, in_parallel_desc, out_parallel_desc));
  const auto& sbp_list = JUST(GetSbpList(out_nd_sbp));
  const auto& output_tensor = JUST(one::functional::ToConsistent(
      partial_sum_input, out_parallel_desc, *sbp_list, GetNoneSbpList()));
  CHECK_OR_RETURN(output_tensor->is_consistent());
  const auto& tensor_placement = JUST(output_tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == out_parallel_desc);
  return output_tensor;
}

Maybe<one::Tensor> Nccl1ToPBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_EQ_OR_RETURN(in_parallel_desc->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsPartialSumNdSbp(out_nd_sbp));
  int64_t root = JUST(out_parallel_desc->MachineId4ParallelId(0));
  std::shared_ptr<one::Tensor> local_tensor = JUST(input->cur_rank_phy_tensor());
  if (root == GlobalProcessCtx::Rank()) {
    // do nothing
  } else {
    std::string device_type = Device::Type4DeviceTag(JUST(input->parallel_desc())->device_tag());
    local_tensor = JUST(one::functional::Constant(*input->shape(), 0, input->dtype(),
                                                  JUST(Device::New(device_type))));
  }
  const auto& output_tensor = JUST(one::functional::ToConsistent(
      local_tensor, out_parallel_desc, *JUST(GetSbpList(out_nd_sbp)), GetNoneSbpList()));
  CHECK_OR_RETURN(output_tensor->is_consistent());
  const auto& tensor_placement = JUST(output_tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == out_parallel_desc);
  return output_tensor;
}

Maybe<one::Tensor> Nccl1ToSBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_EQ_OR_RETURN(in_parallel_desc->parallel_num(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsSplitNdSbp(out_nd_sbp, 0));
  Symbol<cfg::NdSbp> partial_sum_nd_sbp = JUST(GetPartialSumNdSbp());
  static std::shared_ptr<EagerBoxingInterpreter> nccl_1top_boxing_interpreter =
      std::make_shared<Nccl1ToPBoxingInterpreter>();
  const auto& partial_sum_input = JUST(nccl_1top_boxing_interpreter->Interpret(
      input, in_nd_sbp, partial_sum_nd_sbp, in_parallel_desc, out_parallel_desc));
  const auto& sbp_list = JUST(GetSbpList(out_nd_sbp));
  const auto& output_tensor = JUST(one::functional::ToConsistent(
      partial_sum_input, out_parallel_desc, *sbp_list, GetNoneSbpList()));
  CHECK_OR_RETURN(output_tensor->is_consistent());
  const auto& tensor_placement = JUST(output_tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == out_parallel_desc);
  return output_tensor;
}
}  // namespace oneflow
