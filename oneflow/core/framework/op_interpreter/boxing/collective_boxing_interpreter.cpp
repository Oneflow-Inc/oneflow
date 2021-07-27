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
#include "oneflow/core/framework/op_interpreter/boxing/collective_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {

Maybe<void> NcclCollectiveAllGatherBoxingInterpreter::Interpret(
    const one::TensorTuple& inputs, one::TensorTuple* outputs,
    Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBoxingS2B(
      in_parallel_distribution->sbp_parallel(0), out_parallel_distribution->sbp_parallel(0)));
  CHECK_EQ_OR_RETURN(in_parallel_desc, out_parallel_desc);
  std::shared_ptr<one::UserOpExpr> op_expr =
      JUST(op_expr_helper::EagerNcclAllGather(in_parallel_desc));
  outputs->at(0) = JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, inputs));
  return Maybe<void>::Ok();
}

Maybe<void> NcclCollectiveAllReduceBoxingInterpreter::Interpret(
    const one::TensorTuple& inputs, one::TensorTuple* outputs,
    Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBoxingP2B(
      in_parallel_distribution->sbp_parallel(0), out_parallel_distribution->sbp_parallel(0)));
  CHECK_EQ_OR_RETURN(in_parallel_desc, out_parallel_desc);
  std::shared_ptr<one::UserOpExpr> op_expr =
      JUST(op_expr_helper::EagerNcclAllReduce(in_parallel_desc));
  outputs->at(0) = JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, inputs));
  return Maybe<void>::Ok();
}

Maybe<void> NcclCollectiveReduceScatterBoxingInterpreter::Interpret(
    const one::TensorTuple& inputs, one::TensorTuple* outputs,
    Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(
      (EagerBoxingInterpreterUtil::IsBoxingP2S(in_parallel_distribution->sbp_parallel(0),
                                               out_parallel_distribution->sbp_parallel(0))
       || EagerBoxingInterpreterUtil::IsBoxingB2S(in_parallel_distribution->sbp_parallel(0),
                                                  out_parallel_distribution->sbp_parallel(0))));
  CHECK_EQ_OR_RETURN(in_parallel_desc, out_parallel_desc);
  std::shared_ptr<one::UserOpExpr> op_expr =
      JUST(op_expr_helper::EagerNcclReduceScatter(in_parallel_desc, op_type_));
  outputs->at(0) = JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, inputs));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
