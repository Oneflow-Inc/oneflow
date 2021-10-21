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

#include "oneflow/core/autograd/autograd_function.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {
namespace one {

AutogradFunctionBase::AutogradFunctionBase(const std::string& func_name, const FType& forward_fn,
                                           const FType& backward_fn) {
  op_ = CHECK_JUST(FunctionOpExpr::New(func_name, forward_fn, backward_fn));
}

Maybe<TensorTuple> AutogradFunctionBase::Apply(const TensorTuple& inputs) const {
  // TODO(wyg): construct ctx, do forward and process outputs autograd_meta
  OF_UNIMPLEMENTED();
}

}  // namespace one
}  // namespace oneflow
