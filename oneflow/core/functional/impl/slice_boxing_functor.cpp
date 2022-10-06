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

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

namespace {

bool IsSplitSbp(Symbol<SbpParallel> sbp_parallel) { return sbp_parallel->has_split_parallel(); }

Maybe<one::UserOpExpr> EagerSToB(Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc,
                                 Symbol<SbpParallel> src_sbp, const Shape& shape) {
  return one::OpBuilder("eager_s_to_b", *JUST(UniqueStr("eager_s_to_b")))
      .Input("in")
      .Output("out")
      .Attr<int64_t>("in_split_axis", src_sbp->split_parallel().axis())
      .Attr<std::string>("in_parallel_conf", PbMessage2TxtString(in_parallel_desc->parallel_conf()))
      .Attr<std::string>("out_parallel_conf",
                         PbMessage2TxtString(out_parallel_desc->parallel_conf()))
      .Attr<Shape>("shape", shape)
      .Build();
}

static constexpr auto* CachedEagerSToBOpExpr = DECORATE(&EagerSToB, ThreadLocalCopiable);

Maybe<one::UserOpExpr> EagerPToB(Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc, const Shape& shape) {
  return one::OpBuilder("eager_p_to_b", *JUST(UniqueStr("eager_p_to_b")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("in_parallel_conf", PbMessage2TxtString(in_parallel_desc->parallel_conf()))
      .Attr<std::string>("out_parallel_conf",
                         PbMessage2TxtString(out_parallel_desc->parallel_conf()))
      .Attr<Shape>("shape", shape)
      .Build();
}

static constexpr auto* CachedEagerPToBOpExpr = DECORATE(&EagerPToB, ThreadLocalCopiable);

Maybe<one::UserOpExpr> EagerNaiveSToS(Symbol<ParallelDesc> in_parallel_desc,
                                      Symbol<ParallelDesc> out_parallel_desc,
                                      Symbol<SbpParallel> src_sbp, Symbol<SbpParallel> dst_sbp,
                                      const Shape& shape) {
  return one::OpBuilder("eager_naive_s_to_s", *JUST(UniqueStr("eager_naive_s_to_s")))
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

static constexpr auto* CachedEagerNaiveSToSOpExpr = DECORATE(&EagerNaiveSToS, ThreadLocalCopiable);

Maybe<one::UserOpExpr> EagerBToS(Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc,
                                 Symbol<SbpParallel> dst_sbp, const Shape& shape) {
  return one::OpBuilder("eager_b_to_s", *JUST(UniqueStr("eager_b_to_s")))
      .Input("in")
      .Output("out")
      .Attr<int64_t>("out_split_axis", dst_sbp->split_parallel().axis())
      .Attr<std::string>("in_parallel_conf", PbMessage2TxtString(in_parallel_desc->parallel_conf()))
      .Attr<std::string>("out_parallel_conf",
                         PbMessage2TxtString(out_parallel_desc->parallel_conf()))
      .Attr<Shape>("shape", shape)
      .Build();
}

static constexpr auto* CachedEagerBToSOpExpr = DECORATE(&EagerBToS, ThreadLocalCopiable);

Maybe<one::UserOpExpr> EagerPToS(Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc,
                                 Symbol<SbpParallel> dst_sbp, const Shape& shape) {
  return one::OpBuilder("eager_p_to_s", *JUST(UniqueStr("eager_p_to_s")))
      .Input("in")
      .Output("out")
      .Attr<int64_t>("out_split_axis", dst_sbp->split_parallel().axis())
      .Attr<std::string>("in_parallel_conf", PbMessage2TxtString(in_parallel_desc->parallel_conf()))
      .Attr<std::string>("out_parallel_conf",
                         PbMessage2TxtString(out_parallel_desc->parallel_conf()))
      .Attr<Shape>("shape", shape)
      .Build();
}

static constexpr auto* CachedEagerPToSOpExpr = DECORATE(&EagerPToS, ThreadLocalCopiable);

Maybe<one::UserOpExpr> EagerSToP(Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc,
                                 Symbol<SbpParallel> src_sbp, const Shape& shape) {
  return one::OpBuilder("eager_s_to_p", *JUST(UniqueStr("eager_s_to_p")))
      .Input("in")
      .Output("out")
      .Attr<int64_t>("in_split_axis", src_sbp->split_parallel().axis())
      .Attr<std::string>("in_parallel_conf", PbMessage2TxtString(in_parallel_desc->parallel_conf()))
      .Attr<std::string>("out_parallel_conf",
                         PbMessage2TxtString(out_parallel_desc->parallel_conf()))
      .Attr<Shape>("shape", shape)
      .Build();
}

static constexpr auto* CachedEagerSToPOpExpr = DECORATE(&EagerSToP, ThreadLocalCopiable);

}  // namespace

class EagerSToBFunctor {
 public:
  EagerSToBFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> in_parallel_desc,
                           Symbol<ParallelDesc> out_parallel_desc,
                           const std::vector<Symbol<SbpParallel>>& in_sbp_parallels,
                           const Shape& shape) const {
    Symbol<NdSbp> in_nd_sbp = JUST(GetNdSbp(in_sbp_parallels));
    {
      CHECK_OR_RETURN(x->is_local())
          << Error::RuntimeError() << "input tensors `.is_local` should be true";
      CHECK_OR_RETURN(x->is_eager())
          << Error::RuntimeError() << "input tensors `.is_eager` should be true";
      CHECK_OR_RETURN((in_nd_sbp->sbp_parallel_size() == 1)
                      && IsSplitSbp(in_nd_sbp->sbp_parallel(0)))
          << Error::RuntimeError() << "The input tensor's sbp should be (split, )";
    }
    std::shared_ptr<OpExpr> op_expr = JUST(CachedEagerSToBOpExpr(
        in_parallel_desc, out_parallel_desc, SymbolOf(in_nd_sbp->sbp_parallel(0)), shape));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

class EagerPToBFunctor {
 public:
  EagerPToBFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> in_parallel_desc,
                           Symbol<ParallelDesc> out_parallel_desc, const Shape& shape) const {
    {
      CHECK_OR_RETURN(x->is_local())
          << Error::RuntimeError() << "input tensors `.is_local` should be true";
      CHECK_OR_RETURN(x->is_eager())
          << Error::RuntimeError() << "input tensors `.is_eager` should be true";
    }
    std::shared_ptr<OpExpr> op_expr =
        JUST(CachedEagerPToBOpExpr(in_parallel_desc, out_parallel_desc, shape));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

class EagerNaiveSToSFunctor {
 public:
  EagerNaiveSToSFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> in_parallel_desc,
                           Symbol<ParallelDesc> out_parallel_desc,
                           const std::vector<Symbol<SbpParallel>>& in_sbp_parallels,
                           const std::vector<Symbol<SbpParallel>>& out_sbp_parallels,
                           const Shape& shape) const {
    Symbol<NdSbp> in_nd_sbp = JUST(GetNdSbp(in_sbp_parallels));
    Symbol<NdSbp> out_nd_sbp = JUST(GetNdSbp(out_sbp_parallels));
    {
      CHECK_OR_RETURN(x->is_local())
          << Error::RuntimeError() << "input tensors `.is_local` should be true";
      CHECK_OR_RETURN(x->is_eager())
          << Error::RuntimeError() << "input tensors `.is_eager` should be true";
      CHECK_OR_RETURN((in_nd_sbp->sbp_parallel_size() == 1)
                      && IsSplitSbp(in_nd_sbp->sbp_parallel(0)))
          << Error::RuntimeError() << "The input tensor's sbp should be (split, )";
      CHECK_OR_RETURN((out_nd_sbp->sbp_parallel_size() == 1)
                      && IsSplitSbp(out_nd_sbp->sbp_parallel(0)))
          << Error::RuntimeError() << "The output tensor's sbp should be (split, )";
    }
    std::shared_ptr<OpExpr> op_expr = JUST(CachedEagerNaiveSToSOpExpr(
        in_parallel_desc, out_parallel_desc, SymbolOf(in_nd_sbp->sbp_parallel(0)),
        SymbolOf(out_nd_sbp->sbp_parallel(0)), shape));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

class EagerBToSFunctor {
 public:
  EagerBToSFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> in_parallel_desc,
                           Symbol<ParallelDesc> out_parallel_desc,
                           const std::vector<Symbol<SbpParallel>>& out_sbp_parallels,
                           const Shape& shape) const {
    Symbol<NdSbp> out_nd_sbp = JUST(GetNdSbp(out_sbp_parallels));
    {
      CHECK_OR_RETURN(x->is_local())
          << Error::RuntimeError() << "input tensors `.is_local` should be true";
      CHECK_OR_RETURN(x->is_eager())
          << Error::RuntimeError() << "input tensors `.is_eager` should be true";
      CHECK_OR_RETURN((out_nd_sbp->sbp_parallel_size() == 1)
                      && IsSplitSbp(out_nd_sbp->sbp_parallel(0)))
          << Error::RuntimeError() << "The output tensor's sbp should be (split, )";
    }
    std::shared_ptr<OpExpr> op_expr = JUST(CachedEagerBToSOpExpr(
        in_parallel_desc, out_parallel_desc, SymbolOf(out_nd_sbp->sbp_parallel(0)), shape));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

class EagerPToSFunctor {
 public:
  EagerPToSFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> in_parallel_desc,
                           Symbol<ParallelDesc> out_parallel_desc,
                           const std::vector<Symbol<SbpParallel>>& out_sbp_parallels,
                           const Shape& shape) const {
    Symbol<NdSbp> out_nd_sbp = JUST(GetNdSbp(out_sbp_parallels));
    {
      CHECK_OR_RETURN(x->is_local())
          << Error::RuntimeError() << "input tensors `.is_local` should be true";
      CHECK_OR_RETURN(x->is_eager())
          << Error::RuntimeError() << "input tensors `.is_eager` should be true";
      CHECK_OR_RETURN((out_nd_sbp->sbp_parallel_size() == 1)
                      && IsSplitSbp(out_nd_sbp->sbp_parallel(0)))
          << Error::RuntimeError() << "The output tensor's sbp should be (split, )";
    }
    std::shared_ptr<OpExpr> op_expr = JUST(CachedEagerPToSOpExpr(
        in_parallel_desc, out_parallel_desc, SymbolOf(out_nd_sbp->sbp_parallel(0)), shape));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

class EagerSToPFunctor {
 public:
  EagerSToPFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> in_parallel_desc,
                           Symbol<ParallelDesc> out_parallel_desc,
                           const std::vector<Symbol<SbpParallel>>& in_sbp_parallels,
                           const Shape& shape) const {
    Symbol<NdSbp> in_nd_sbp = JUST(GetNdSbp(in_sbp_parallels));
    {
      CHECK_OR_RETURN(x->is_local())
          << Error::RuntimeError() << "input tensors `.is_local` should be true";
      CHECK_OR_RETURN(x->is_eager())
          << Error::RuntimeError() << "input tensors `.is_eager` should be true";
      CHECK_OR_RETURN((in_nd_sbp->sbp_parallel_size() == 1)
                      && IsSplitSbp(in_nd_sbp->sbp_parallel(0)))
          << Error::RuntimeError() << "The input tensor's sbp should be (split, )";
    }
    std::shared_ptr<OpExpr> op_expr = JUST(CachedEagerSToPOpExpr(
        in_parallel_desc, out_parallel_desc, SymbolOf(in_nd_sbp->sbp_parallel(0)), shape));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::EagerSToBFunctor>("EagerSToB");
  m.add_functor<impl::EagerPToBFunctor>("EagerPToB");
  m.add_functor<impl::EagerNaiveSToSFunctor>("EagerNaiveSToS");
  m.add_functor<impl::EagerBToSFunctor>("EagerBToS");
  m.add_functor<impl::EagerPToSFunctor>("EagerPToS");
  m.add_functor<impl::EagerSToPFunctor>("EagerSToP");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
