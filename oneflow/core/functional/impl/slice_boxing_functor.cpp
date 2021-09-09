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

Maybe<one::UserOpExpr> EagerBToS(Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc, const Shape& shape) {
  return one::OpBuilder("eager_b_to_s", *JUST(UniqueStr("eager_b_to_s")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("in_parallel_conf", PbMessage2TxtString(in_parallel_desc->parallel_conf()))
      .Attr<std::string>("out_parallel_conf",
                         PbMessage2TxtString(out_parallel_desc->parallel_conf()))
      .Attr<Shape>("shape", shape)
      .Build();
}

static constexpr auto* CachedEagerBToSpExpr = DECORATE(&EagerBToS, ThreadLocalCopiable);

Maybe<one::UserOpExpr> EagerPToS(Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc, const Shape& shape) {
  return one::OpBuilder("eager_p_to_s", *JUST(UniqueStr("eager_p_to_s")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("in_parallel_conf", PbMessage2TxtString(in_parallel_desc->parallel_conf()))
      .Attr<std::string>("out_parallel_conf",
                         PbMessage2TxtString(out_parallel_desc->parallel_conf()))
      .Attr<Shape>("shape", shape)
      .Build();
}

static constexpr auto* CachedEagerPToSpExpr = DECORATE(&EagerPToS, ThreadLocalCopiable);

}  // namespace

class EagerBToSFunctor {
 public:
  EagerBToSFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> in_parallel_desc,
                           Symbol<ParallelDesc> out_parallel_desc, const Shape& shape) const {
    {
      CHECK_OR_RETURN(x->is_local());
      CHECK_OR_RETURN(x->is_eager());
      // CHECK_OR_RETURN(!x->is_cuda());
    }
    std::shared_ptr<OpExpr> op_expr =
        JUST(CachedEagerBToSpExpr(in_parallel_desc, out_parallel_desc, shape));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

class EagerPToSFunctor {
 public:
  EagerPToSFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> in_parallel_desc,
                           Symbol<ParallelDesc> out_parallel_desc, const Shape& shape) const {
    {
      CHECK_OR_RETURN(x->is_local());
      CHECK_OR_RETURN(x->is_eager());
    }
    std::shared_ptr<OpExpr> op_expr =
        JUST(CachedEagerPToSpExpr(in_parallel_desc, out_parallel_desc, shape));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::EagerBToSFunctor>("EagerBToS");
  m.add_functor<impl::EagerPToSFunctor>("EagerPToS");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
