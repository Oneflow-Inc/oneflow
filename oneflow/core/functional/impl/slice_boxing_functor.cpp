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
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/scalar.h"
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

namespace {

bool IsSplitSbp(Symbol<cfg::SbpParallel> sbp_parallel) {
  return sbp_parallel->has_split_parallel();
}

Maybe<one::UserOpExpr> EagerS0ToS0(Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc,
                                   Symbol<cfg::SbpParallel> src_sbp,
                                   Symbol<cfg::SbpParallel> dst_sbp, const Shape& shape) {
  return one::OpBuilder("eager_s0_to_s0", *JUST(UniqueStr("eager_s0_to_s0")))
      .Input("in")
      .Output("out")
      .Attr<int64_t>("in_split_axis", src_sbp->split_parallel().axis())
      .Attr<int64_t>("out_split_axis", dst_sbp->split_parallel().axis())
      .Attr<std::string>("in_parallel_conf", PbMessage2TxtString(in_parallel_desc->parallel_conf()))
      .Attr<std::string>("out_parallel_conf",
                         PbMessage2TxtString(out_parallel_desc->parallel_conf()))
      .Attr<Shape>("shape", shape)
      .Build();
}

static constexpr auto* CachedEagerS0ToS0OpExpr = DECORATE(&EagerS0ToS0, ThreadLocalCopiable);

}  // namespace

class EagerS0ToS0Functor {
 public:
  EagerS0ToS0Functor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> in_parallel_desc,
                           Symbol<ParallelDesc> out_parallel_desc,
                           const std::vector<Symbol<cfg::SbpParallel>>& in_sbp_parallels,
                           const std::vector<Symbol<cfg::SbpParallel>>& out_sbp_parallels,
                           const Shape& shape) const {
    Symbol<cfg::NdSbp> in_nd_sbp = JUST(GetNdSbp(in_sbp_parallels));
    Symbol<cfg::NdSbp> out_nd_sbp = JUST(GetNdSbp(out_sbp_parallels));
    {
      CHECK_OR_RETURN(x->is_local());
      CHECK_OR_RETURN(x->is_eager());
      CHECK_OR_RETURN(!x->is_cuda());
      CHECK_EQ_OR_RETURN(in_nd_sbp->sbp_parallel_size(), 1);
      CHECK_OR_RETURN(IsSplitSbp(in_nd_sbp->sbp_parallel(0)));
      CHECK_EQ_OR_RETURN(out_nd_sbp->sbp_parallel_size(), 1);
      CHECK_OR_RETURN(IsSplitSbp(out_nd_sbp->sbp_parallel(0)));
    }
    std::shared_ptr<OpExpr> op_expr = JUST(CachedEagerS0ToS0OpExpr(
        in_parallel_desc, out_parallel_desc, SymbolOf(in_nd_sbp->sbp_parallel(0)),
        SymbolOf(out_nd_sbp->sbp_parallel(0)), shape));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::EagerS0ToS0Functor>("EagerS0ToS0"); };

}  // namespace functional
}  // namespace one
}  // namespace oneflow
