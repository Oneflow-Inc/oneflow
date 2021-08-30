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
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/collective_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"

namespace oneflow {

namespace {

Maybe<one::UserOpExpr> EagerNcclAllReduce(Symbol<ParallelDesc> parallel_desc) {
  return one::OpBuilder("eager_nccl_all_reduce", *JUST(UniqueStr("eager_nccl_all_reduce")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Build();
}

auto* CachedEagerNcclAllReduceOpExpr = DECORATE(&EagerNcclAllReduce, ThreadLocal);

Maybe<one::UserOpExpr> EagerNcclAllGather(Symbol<ParallelDesc> parallel_desc) {
  return one::OpBuilder("eager_nccl_all_gather", *JUST(UniqueStr("eager_nccl_all_gather")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Build();
}

auto* CachedEagerNcclAllGatherOpExpr = DECORATE(&EagerNcclAllGather, ThreadLocal);

Maybe<one::UserOpExpr> EagerNcclReduceScatter(Symbol<ParallelDesc> parallel_desc,
                                              const std::string& op_type) {
  return one::OpBuilder("eager_nccl_reduce_scatter", *JUST(UniqueStr("eager_nccl_reduce_scatter")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Attr<std::string>("op_type", op_type)
      .Build();
}
auto* CachedNcclReduceScatterOpExpr = DECORATE(&EagerNcclReduceScatter, ThreadLocalCopiable);

Maybe<one::UserOpExpr> EagerNcclS2S(Symbol<ParallelDesc> parallel_desc,
                                    Symbol<cfg::SbpParallel> src_sbp,
                                    Symbol<cfg::SbpParallel> dst_sbp) {
  return one::OpBuilder("eager_nccl_s2s", *JUST(UniqueStr("eager_nccl_s2s")))
      .Input("in")
      .Output("out")
      .Attr<int64_t>("in_split_axis", src_sbp->split_parallel().axis())
      .Attr<int64_t>("out_split_axis", dst_sbp->split_parallel().axis())
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Build();
}

auto* CachedEagerNcclS2SOpExpr = DECORATE(&EagerNcclS2S, ThreadLocal);

}  // namespace

Maybe<one::Tensor> NcclCollectiveAllGatherBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBoxingS2B(in_nd_sbp->sbp_parallel(0),
                                                          out_nd_sbp->sbp_parallel(0)));
  CHECK_OR_RETURN(in_parallel_desc == out_parallel_desc);
  const auto& op_expr = JUST(CachedEagerNcclAllGatherOpExpr(in_parallel_desc));
  return JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {input}));
}

Maybe<one::Tensor> NcclCollectiveAllReduceBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBoxingP2B(in_nd_sbp->sbp_parallel(0),
                                                          out_nd_sbp->sbp_parallel(0)));
  CHECK_OR_RETURN(in_parallel_desc == out_parallel_desc);
  const auto& op_expr = JUST(CachedEagerNcclAllReduceOpExpr(in_parallel_desc));
  return JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {input}));
}

Maybe<one::Tensor> NcclCollectiveReduceScatterBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN((EagerBoxingInterpreterUtil::IsBoxingP2S(in_nd_sbp->sbp_parallel(0),
                                                           out_nd_sbp->sbp_parallel(0))
                   || EagerBoxingInterpreterUtil::IsBoxingB2S(in_nd_sbp->sbp_parallel(0),
                                                              out_nd_sbp->sbp_parallel(0))));
  CHECK_OR_RETURN(in_parallel_desc == out_parallel_desc);
  const auto& op_expr = JUST(CachedNcclReduceScatterOpExpr(in_parallel_desc, op_type_));
  return JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {input}));
}

Maybe<one::Tensor> NcclCollectiveS2SBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBoxingS2S(in_nd_sbp->sbp_parallel(0),
                                                          out_nd_sbp->sbp_parallel(0)));
  CHECK_OR_RETURN(in_parallel_desc == out_parallel_desc);
  const auto& op_expr =
      JUST(CachedEagerNcclS2SOpExpr(in_parallel_desc, SymbolOf(in_nd_sbp->sbp_parallel(0)),
                                    SymbolOf(out_nd_sbp->sbp_parallel(0))));
  return JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {input}));
}
}  // namespace oneflow
