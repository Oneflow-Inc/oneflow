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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr_grad_function.h"

namespace oneflow {
namespace one {

/*static*/ Maybe<TensorTuple> AutogradFunctionBase::Apply(const std::string& name,
                                                          const FType& forward_fn,
                                                          const FType& backward_fn,
                                                          const TensorTuple& inputs) {
  std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>();
  const auto& op = JUST(FunctionOpExpr::New(name, forward_fn, backward_fn));
  JUST(OpInterpUtil::Dispatch(*op, inputs, outputs.get(), {}));
  const HashSet<Tensor*>& non_differentiable_tensors = op->state()->NonDifferentiableTensors();
  for (const auto& tensor : *outputs) {
    if (non_differentiable_tensors.find(tensor.get()) != non_differentiable_tensors.end()) {
      JUST(tensor->set_requires_grad(false));
    }
  }
  return outputs;
}

}  // namespace one
}  // namespace oneflow
