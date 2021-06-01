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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/scalar.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ConstantFunctor {
 public:
  ConstantFunctor() { op_ = CHECK_JUST(one::OpBuilder("constant").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Scalar& value, const DataType& dtype) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype));
    if (IsIntegralDataType(dtype)) {
      JUST(attrs.SetAttr<bool>("is_floating_value", false));
      JUST(attrs.SetAttr<int64_t>("integer_value", JUST(value.As<int64_t>())));
    } else {
      JUST(attrs.SetAttr<bool>("is_floating_value", true));
      JUST(attrs.SetAttr<double>("floating_value", JUST(value.As<double>())));
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ZerosLikeFunctor : public UnaryFunctor {
 public:
  ZerosLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("zero_like").Input("like").Output("out").Build());
  }
};

class OnesLikeFunctor : public UnaryFunctor {
 public:
  OnesLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("ones_like").Input("like").Output("out").Build());
  }
};

class FlattenFunctor {
 public:
  FlattenFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("flatten").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& start_dim,
                           const int32_t& end_dim) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("start_dim", start_dim));
    JUST(attrs.SetAttr<int32_t>("end_dim", end_dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ArgWhereFunctor {
 public:
  ArgWhereFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("argwhere").Input("input").Output("output").Output("output_size").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const DataType& dtype) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<DataType>("dtype", dtype));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastLikeFunctor {
 public:
  BroadcastLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_like").Input("x").Input("like").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& like,
                           const std::vector<int32_t>& broadcast_axes) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("broadcast_axes", broadcast_axes));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, like}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConcatFunctor {
 public:
  ConcatFunctor() {
    ops_.resize(kMaxInputCount);
    for (int n = 2; n < ops_.size(); ++n) {
      ops_[n] = CHECK_JUST(one::OpBuilder("concat").Input("in", n).Output("out").Build());
    }
  }
  Maybe<Tensor> operator()(const TensorTuple& inputs, const int64_t& axis,
                           const int64_t& max_dim_size) const {
    CHECK_GE_OR_RETURN(inputs.size(), 2);
    CHECK_LT_OR_RETURN(inputs.size(), ops_.size())
        << "The maximum number supported of inputs is " << ops_.size();
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", axis));
    JUST(attrs.SetAttr<int64_t>("max_dim_size", max_dim_size));
    return OpInterpUtil::Dispatch<Tensor>(*ops_.at(inputs.size()), inputs, attrs);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> ops_;
};

class ExpandFunctor {
 public:
  ExpandFunctor() { op_ = CHECK_JUST(one::OpBuilder("expand").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& in_shape,
                           const std::vector<int32_t>& out_shape,
                           const std::vector<int32_t>& stride) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("in_shape", in_shape));
    JUST(attrs.SetAttr<std::vector<int32_t>>("out_shape", out_shape));
    JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GatherFunctor {
 public:
  GatherFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("gather").Input("in").Input("indices").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& indices, const int64_t& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, indices}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimGatherFunctor {
 public:
  DimGatherFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("dim_gather").Input("input").Input("index").Output("output").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& indices, const int32_t& dim) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, indices}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ConstantFunctor>("Constant");
  m.add_functor<impl::ZerosLikeFunctor>("ZerosLike");
  m.add_functor<impl::OnesLikeFunctor>("OnesLike");
  m.add_functor<impl::FlattenFunctor>("Flatten");
  m.add_functor<impl::ArgWhereFunctor>("ArgWhere");
  m.add_functor<impl::BroadcastLikeFunctor>("BroadcastLike");
  m.add_functor<impl::ConcatFunctor>("Concat");
  m.add_functor<impl::ExpandFunctor>("Expand");
  m.add_functor<impl::GatherFunctor>("Gather");
  m.add_functor<impl::DimGatherFunctor>("DimGather");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
