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

#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/impl/common.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ThrowErrorFunctor final {
 public:
  ThrowErrorFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("throw_error").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input) const {
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {input}));
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

using namespace impl;

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<ThrowErrorFunctor>("ThrowError"); };

}  // namespace functional
}  // namespace one
}  // namespace oneflow
