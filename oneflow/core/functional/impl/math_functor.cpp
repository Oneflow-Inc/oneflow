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

class AddNFunctor {
 public:
  AddNFunctor() {
    op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 1; n < op_.size(); ++n) {
      op_[n] = CHECK_JUST(one::OpBuilder("add_n").Input("in", n + 1).Output("out").Build());
    }
  }
  Maybe<Tensor> operator()(const TensorTuple& inputs) const {
    CHECK_GE_OR_RETURN(inputs.size(), 2);
    TensorTuple outputs;
    for (int i = 0; i < inputs.size(); i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < inputs.size() ? kMaxInputCount : inputs.size() - i;
      TensorTuple partial_inputs(size);
      std::copy(inputs.begin() + i, inputs.begin() + i + size, partial_inputs.begin());
      outputs.push_back(JUST(OpInterpUtil::Dispatch<Tensor>(*op_.at(size - 1), partial_inputs)));
    }
    if (outputs.size() == 1) { return outputs.at(0); }
    return this->operator()(outputs);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> op_;
};

class ScalarAddFunctor {
 public:
  ScalarAddFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scalar_add").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in ScalarAdd shoule be float or int.";
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScalarMulFunctor {
 public:
  ScalarMulFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scalar_mul").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in ScalarMul shoule be float or int.";
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScalarPowFunctor {
 public:
  ScalarPowFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scalar_pow").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("exponent", JUST(scalar.As<double>())));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReduceSumFunctor {
 public:
  ReduceSumFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("reduce_sum").Input("input_tensor").Output("output_tensor").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis,
                           const bool& keepdims) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    JUST(attrs.SetAttr<bool>("keepdims", keepdims));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TransposeFunctor {
 public:
  TransposeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("transpose").Input("input").Output("output").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& permute) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("perm", permute));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RangeFunctor {
 public:
  RangeFunctor() { op_ = CHECK_JUST(one::OpBuilder("range").Output("out").Build()); }
  Maybe<Tensor> operator()(const int64_t& start, const int64_t& limit, const int64_t& delta,
                           const DataType& dtype) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("start", start));
    JUST(attrs.SetAttr<int64_t>("limit", limit));
    JUST(attrs.SetAttr<int64_t>("delta", delta));
    JUST(attrs.SetAttr<DataType>("dtype", dtype));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ArgMaxFunctor : public UnaryFunctor {
 public:
  ArgMaxFunctor() { op_ = CHECK_JUST(one::OpBuilder("argmax").Input("in").Output("out").Build()); }
};

class CastFunctor {
 public:
  CastFunctor() { op_ = CHECK_JUST(one::OpBuilder("cast").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const DataType& dtype) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<DataType>("dtype", dtype));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::AddNFunctor>("AddN");
  m.add_functor<impl::ScalarAddFunctor>("ScalarAdd");
  m.add_functor<impl::ScalarMulFunctor>("ScalarMul");
  m.add_functor<impl::ScalarPowFunctor>("ScalarPow");
  m.add_functor<impl::ReduceSumFunctor>("ReduceSum");
  m.add_functor<impl::TransposeFunctor>("Transpose");
  m.add_functor<impl::RangeFunctor>("Range");
  m.add_functor<impl::ArgMaxFunctor>("ArgMax");
  m.add_functor<impl::CastFunctor>("Cast");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
