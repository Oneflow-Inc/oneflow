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

#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/tensor_arg.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor.h"

namespace oneflow {
namespace one {

void StackFunctionNode::ReleaseOutTensorArgs() {
  for (std::shared_ptr<TensorArg> tensor_arg : out_grads_) { tensor_arg->Release(); }
}

void StackFunctionNode::ReleaseGraph() {
  inputs_.reset();
  outputs_.reset();
  in_grads_.clear();
  out_grads_.clear();
  backward_fn_.reset();
}

Maybe<void> StackFunctionNode::Apply(bool create_graph) {
  TODO();  // wangyinggang: run backward_fn
  return Maybe<void>::Ok();
}

Maybe<TensorTuple> StackAutogradEngine::Execute(const TensorTuple& outputs,
                                                const TensorTuple& inputs,
                                                const TensorTuple& out_grads, bool retain_graph,
                                                bool create_graph) {
  std::shared_ptr<TensorTuple> captured_tensors = std::make_shared<TensorTuple>(inputs.size());
  TODO();  // wangyinggang: run each FunctionNode and capture input grads
  return captured_tensors;
}

const std::shared_ptr<FunctionNode>& StackAutogradEngine::AddBackwardFuncPtr(
    const std::shared_ptr<const std::function<void()>>& backward_fn, const TensorTuple& inputs,
    const TensorTuple& outputs) {
  std::shared_ptr<FunctionNode> func_node = std::make_shared<StackFunctionNode>();
  for (std::shared_ptr<Tensor> out_tensor : outputs) {
    TODO();  // setter grad_fn for output tensors
  }
  return std::move(func_node);
}

}  // namespace one
}  // namespace oneflow
