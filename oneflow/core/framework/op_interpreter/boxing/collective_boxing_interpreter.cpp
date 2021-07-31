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
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/collective_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"

namespace oneflow {

namespace {

Maybe<one::UserOpExpr> EagerNcclAllReduce(Symbol<ParallelDesc> parallel_desc) {
  return one::OpBuilder("eager_nccl_all_reduce", *CHECK_JUST(UniqueStr("eager_nccl_all_reduce")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Build();
}

Maybe<one::UserOpExpr> EagerNcclAllGather(Symbol<ParallelDesc> parallel_desc) {
  return one::OpBuilder("eager_nccl_all_gather", *CHECK_JUST(UniqueStr("eager_nccl_all_gather")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Build();
}

Maybe<one::UserOpExpr> EagerNcclReduceScatter(Symbol<ParallelDesc> parallel_desc,
                                              const std::string& op_type) {
  return one::OpBuilder("eager_nccl_reduce_scatter",
                        *CHECK_JUST(UniqueStr("eager_nccl_reduce_scatter")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Attr<std::string>("op_type", op_type)
      .Build();
}

Maybe<one::UserOpExpr> FindOrCreatEagerNcclNcclAllGatherOpExpr(Symbol<ParallelDesc> parallel_desc) {
  thread_local HashMap<Symbol<ParallelDesc>, std::shared_ptr<one::UserOpExpr>>
      parallel_desc2eager_nccl_all_gather;
  auto iter = parallel_desc2eager_nccl_all_gather.find(parallel_desc);
  if (iter == parallel_desc2eager_nccl_all_gather.end()) {
    std::shared_ptr<one::UserOpExpr> op_expr = JUST(EagerNcclAllGather(parallel_desc));
    iter = parallel_desc2eager_nccl_all_gather.emplace(parallel_desc, op_expr).first;
  }
  return iter->second;
}

Maybe<one::UserOpExpr> FindOrCreatEagerNcclNcclAllReduceOpExpr(Symbol<ParallelDesc> parallel_desc) {
  thread_local HashMap<Symbol<ParallelDesc>, std::shared_ptr<one::UserOpExpr>>
      parallel_desc2eager_nccl_all_reduce;
  auto iter = parallel_desc2eager_nccl_all_reduce.find(parallel_desc);
  if (iter == parallel_desc2eager_nccl_all_reduce.end()) {
    std::shared_ptr<one::UserOpExpr> op_expr = JUST(EagerNcclAllReduce(parallel_desc));
    iter = parallel_desc2eager_nccl_all_reduce.emplace(parallel_desc, op_expr).first;
  }
  return iter->second;
}

Maybe<one::UserOpExpr> FindOrCreatEagerNcclNcclReduceScatterOpExpr(
    Symbol<ParallelDesc> parallel_desc, const std::string& op_type) {
  thread_local HashMap<std::pair<Symbol<ParallelDesc>, std::string>,
                       std::shared_ptr<one::UserOpExpr>>
      parallel_desc_and_reduce_type2eager_nccl_all_gather;
  const auto& key = std::make_pair(parallel_desc, op_type);
  auto iter = parallel_desc_and_reduce_type2eager_nccl_all_gather.find(key);
  if (iter == parallel_desc_and_reduce_type2eager_nccl_all_gather.end()) {
    std::shared_ptr<one::UserOpExpr> op_expr = JUST(EagerNcclReduceScatter(parallel_desc, op_type));
    iter = parallel_desc_and_reduce_type2eager_nccl_all_gather.emplace(key, op_expr).first;
  }
  return iter->second;
}
}  // namespace

Maybe<void> NcclCollectiveAllGatherBoxingInterpreter::Interpret(
    const one::TensorTuple& inputs, one::TensorTuple* outputs,
    Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBoxingS2B(
      in_parallel_distribution->sbp_parallel(0), out_parallel_distribution->sbp_parallel(0)));
  CHECK_EQ_OR_RETURN(in_parallel_desc, out_parallel_desc);
  std::shared_ptr<one::UserOpExpr> op_expr =
      JUST(FindOrCreatEagerNcclNcclAllGatherOpExpr(in_parallel_desc));
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
      JUST(FindOrCreatEagerNcclNcclAllReduceOpExpr(in_parallel_desc));
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
      JUST(FindOrCreatEagerNcclNcclReduceScatterOpExpr(in_parallel_desc, op_type_));
  outputs->at(0) = JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, inputs));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
