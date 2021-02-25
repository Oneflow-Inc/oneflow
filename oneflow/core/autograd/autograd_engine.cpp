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
#include "oneflow/core/framework/tensor.h"

namespace oneflow {

namespace one {

void StackFunctionNode::ReleaseGraph() {
  inputs_->clear();
  outputs_->clear();
  in_grads_.clear();
  out_grads_.clear();
  backward_fn_.reset();
}

void StackFunctionNode::ReleaseOutTensorArgs() {
  // TODO: Release tensor_args
  // for (auto& out_grad : out_grads_) { out_grad->Release(); }
}

void StackFunctionNode::Apply(bool create_graph) {
  if (!backward_fn_) {
    // TODO: use maybe catch error
    std::cout << "RuntimeError: Trying to backward through the graph a second time, but the saved "
                 "intermediate results have already been freed. Specify retain_graph=True when "
                 "calling backward the first time."
              << std::endl;
    return;
  }
  // Skips this node when all out_grads not ready
  // TODO: check all tensor_arg is empty
  // if (std::all_of(out_grads_.begin(), out_grads_.end(),
  //                 [](const std::shared_ptr<TensorArg>& tensor_arg) { return tensor_arg->empty();
  //                 }))
  //   return;

  // Inits empty `out_grads_` to zeros
  // auto out_grads = std::make_shared<TensorList>();
  // for (int i = 0; i < out_grads_.size(); i++) {
  //   if (out_grads_[i]->empty()) { out_grads_[i]->init_zeros_like(outputs_->at(i)); }
  //   out_grads->push_back(out_grads_[i]->get_tensor_ptr());
  // }

  // TODO: calls `backward_fn_`
  // backward_fn_(out_grads, inputs_, outputs_, create_graph);
}

std::shared_ptr<TensorList> StackAutogradEngine::Execute(
    const std::shared_ptr<TensorList>& outputs, const std::shared_ptr<TensorList>& inputs,
    const std::shared_ptr<TensorList>& out_grads, bool retain_graph, bool create_graph) {
  auto capture_tensors = std::make_shared<TensorList>(inputs->size());
  // TODO: calls FunctionNode in list one by one and capture each input grad
  return capture_tensors;
}

// TODO: const std::shared_ptr<const FunctionNode>& StackAutogradEngine::AddBackwardFuncPtr(...) {}

}  // namespace one

}  // namespace oneflow
