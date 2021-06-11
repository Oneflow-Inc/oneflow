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
#include "oneflow/core/functional/scalar.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class AddFunctor {
 public:
  AddFunctor() {
    add_op_ = CHECK_JUST(one::OpBuilder("add_n").Input("in", 2).Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<Tensor>(*add_op_, {x, y});
  }

 private:
  std::shared_ptr<OpExpr> add_op_;
};

class AddNFunctor {
 public:
  AddNFunctor() {
    add_n_op_.resize(128 /*the maximum number of inputs*/);
    for (int n = 2; n < add_n_op_.size(); ++n) {
      add_n_op_[n] = CHECK_JUST(one::OpBuilder("add_n").Input("in", n).Output("out").Build());
    }
  }
  Maybe<Tensor> operator()(const TensorTuple& inputs) const {
    CHECK_GE_OR_RETURN(inputs.size(), 2);
    CHECK_LT_OR_RETURN(inputs.size(), add_n_op_.size())
        << "The maximum number supported of inputs is " << add_n_op_.size();
    return OpInterpUtil::Dispatch<Tensor>(*add_n_op_.at(inputs.size()), inputs);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> add_n_op_;
};

class AddScalarFunctor {
 public:
  AddScalarFunctor() {
    add_scalar_op_ = CHECK_JUST(one::OpBuilder("scalar_add").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      return OpInterpUtil::Dispatch<Tensor>(*add_scalar_op_, {x}, attrs);
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      return OpInterpUtil::Dispatch<Tensor>(*add_scalar_op_, {x}, attrs);
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }

 private:
  std::shared_ptr<OpExpr> add_scalar_op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::AddFunctor>("Add");
  m.add_functor<impl::AddNFunctor>("AddN");
  m.add_functor<impl::AddScalarFunctor>("AddScalar");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
