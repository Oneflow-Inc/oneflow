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

#include "oneflow/core/functional/functor_0.h"
#include "oneflow/core/functional/functor.h"
#include "oneflow/core/functional/function_library.h"

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

AddFunctor::AddFunctor() { add_op_ = CHECK_JUST(op_expr_helper::AddOp()); }

Maybe<Tensor> AddFunctor::operator()(const std::shared_ptr<one::Tensor>& x,
                                     const std::shared_ptr<one::Tensor>& y) const {
  return OpInterpUtil::Dispatch<Tensor>(*add_op_, {x, y});
}

AddNFunctor::AddNFunctor() {
  add_n_op_.resize(128 /*the maximum number of inputs*/);
  for (int n = 2; n < add_n_op_.size(); ++n) {
    add_n_op_[n] = CHECK_JUST(op_expr_helper::AddNOp(n));
  }
}

Maybe<Tensor> AddNFunctor::operator()(const TensorTuple& inputs) const {
  CHECK_GE_OR_RETURN(inputs.size(), 2);
  CHECK_LT_OR_RETURN(inputs.size(), add_n_op_.size())
      << "The maximum number supported of inputs is " << add_n_op_.size();
  return OpInterpUtil::Dispatch<Tensor>(*add_n_op_.at(inputs.size()), inputs);
}

AddScalarFunctor::AddScalarFunctor() {
  add_scalar_op_ = CHECK_JUST(op_expr_helper::ScalarAddOp<float>(0.f));
}

Maybe<Tensor> AddScalarFunctor::operator()(const std::shared_ptr<one::Tensor>& x,
                                           const Scalar& scalar) const {
  MutableAttrMap attrs;
  if (scalar.IsFloatingPoint()) {
    attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>()));
    attrs.SetAttr<bool>("has_float_operand", true);
    attrs.SetAttr<bool>("has_int_operand", false);
    return OpInterpUtil::Dispatch<Tensor>(*add_scalar_op_, {x}, attrs);
  } else if (scalar.IsIntegral()) {
    attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>()));
    attrs.SetAttr<bool>("has_float_operand", false);
    attrs.SetAttr<bool>("has_int_operand", true);
    return OpInterpUtil::Dispatch<Tensor>(*add_scalar_op_, {x}, attrs);
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
}

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::AddFunctor>("add");
  m.add_functor<impl::AddNFunctor>("add_n");
  m.add_functor<impl::AddScalarFunctor>("add_scalar");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
