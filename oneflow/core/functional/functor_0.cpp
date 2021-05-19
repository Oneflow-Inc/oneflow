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

#include "oneflow/core/functional/functor.h"
#include "oneflow/core/functional/functor_library.h"

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {
namespace functional {

class AddScalarFunctor {
 public:
  AddScalarFunctor() {
    float_op_ = CHECK_JUST(op_expr_helper::ScalarAddOp<float>(/*scalar=*/0.f));
    integer_op_ = CHECK_JUST(op_expr_helper::ScalarAddOp<int32_t>(/*scalar=*/0));
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<cfg::AttrValue>& scalar) const {
    const OpExpr* op = nullptr;
    MutableCfgAttrMap attrs;
    switch (scalar->value_case()) {
      case kAtFloat:
        op = float_op_.get();
        attrs.SetAttr("float_operand", scalar);
        break;
      case kAtInt32:
        op = integer_op_.get();
        attrs.SetAttr("int_operand", scalar);
        break;
      default: UNIMPLEMENTED_THEN_RETURN();
    }
    return OpInterpUtil::Dispatch<Tensor>(*op, {a}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> float_op_;
  std::shared_ptr<OpExpr> integer_op_;
};

class AddFunctor {
 public:
  AddFunctor() { op_ = CHECK_JUST(op_expr_helper::AddOp()); }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {a, b});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

ONEFLOW_FUNCTOR_LIBRARY(m) {
  m.add_functor<AddScalarFunctor>("add_scalar");
  m.add_functor<AddFunctor>("add");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
