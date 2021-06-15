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
#include "oneflow/core/functional/function_library.h"

namespace oneflow {

namespace one {

namespace functional {

namespace impl {

class ReLUFunctor {
 public:
  ReLUFunctor() {
    relu_op_ = CHECK_JUST(one::OpBuilder("relu").Input("in", 1).Output("out", 1).Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x) const {
    return OpInterpUtil::Dispatch<Tensor>(*relu_op_, {x});
  }

 private:
  std::shared_ptr<OpExpr> relu_op_;
};

class ReLUInplaceFunctor {
 public:
  ReLUInplaceFunctor() {
    relu_op_ = CHECK_JUST(one::OpBuilder("relu").Input("in", 1).Output("out", 1).Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x) const {
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    JUST(JUST(OpInterpUtil::GetInterpreter())->Apply(*relu_op_, {x}, outputs.get(), {}));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> relu_op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ReLUFunctor>("relu");
  m.add_functor<impl::ReLUInplaceFunctor>("relu_");
};
}  // namespace functional
}  // namespace one
}  // namespace oneflow
