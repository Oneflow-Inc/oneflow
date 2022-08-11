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
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/boxing/eager_boxing_interpreter.h"

namespace oneflow {

namespace {

bool RawIsSplitSbp(Symbol<SbpParallel> sbp_parallel) { return sbp_parallel->has_split_parallel(); }

static constexpr auto* IsSplitSbp = DECORATE(&RawIsSplitSbp, ThreadLocalCached);

bool RawIsPartialSumSbp(Symbol<SbpParallel> sbp_parallel) {
  return sbp_parallel->has_partial_sum_parallel();
}

static constexpr auto* IsPartialSumSbp = DECORATE(&RawIsPartialSumSbp, ThreadLocalCached);

Maybe<one::UserOpExpr> EagerSymmetricSToP(Symbol<ParallelDesc> parallel_desc,
                                          Symbol<SbpParallel> src_sbp, const Shape& logical_shape) {
  return one::OpBuilder("eager_symmetric_s_to_p", *JUST(UniqueStr("eager_symmetric_s_to_p")))
      .Input("in")
      .Output("out")
      .Attr<int64_t>("in_split_axis", src_sbp->split_parallel().axis())
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Build();
}

static constexpr auto* CachedEagerSymmetricSToPOpExpr =
    DECORATE(&EagerSymmetricSToP, ThreadLocalCachedCopiable);

// NOLINTBEGIN(maybe-need-error-msg)
Maybe<void> RawCheckSymmetricSToP(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                  const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);

  CHECK_OR_RETURN(IsSplitSbp(in->nd_sbp()->sbp_parallel(0)));
  CHECK_OR_RETURN(IsPartialSumSbp(out->nd_sbp()->sbp_parallel(0)));

  CHECK_OR_RETURN(in->placement() == out->placement());
  return Maybe<void>::Ok();
}
// NOLINTEND(maybe-need-error-msg)

static constexpr auto* CheckSymmetricSToP =
    DECORATE(&RawCheckSymmetricSToP, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> SymmetricSToP(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
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

  std::shared_ptr<one::OpExpr> op_expr = JUST(CachedEagerSymmetricSToPOpExpr(
      tensor_placement, SymbolOf(tensor_nd_sbp->sbp_parallel(0)), *tensor->shape()));

  return JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {tensor}));
}

COMMAND(RegisterBoxingFunction("symmetric-s-to-p", CheckSymmetricSToP, &SymmetricSToP));

}  // namespace oneflow
