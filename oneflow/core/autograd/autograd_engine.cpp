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

void StackFunctionNode::ReleaseTensorGrads() {
  for (auto& out_grad : out_grads_) { out_grad->reset(); }
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
  // skip when all out_grads not ready
  if (std::all_of(out_grads_.begin(), out_grads_.end(),
                  [](const std::shared_ptr<TensorArg>& tensor_arg) { return tensor_arg->empty(); }))
    return;

  auto out_grads = std::make_shared<TensorList>();
  for (int i = 0; i < out_grads_.size(); i++) {
    if (out_grads_[i]->empty()) { out_grads_[i]->init_zeros_like(outputs_->at(i)); }
    out_grads->push_back(out_grads_[i]->get_tensor_ptr());
  }
  // TODO: according to backward_fn interface
  // backward_fn_(out_grads, inputs_, outputs_, create_graph);
}

std::shared_ptr<TensorList> StackAutogradEngine::Execute(
    const std::shared_ptr<TensorList>& outputs, const std::shared_ptr<TensorList>& inputs,
    const std::shared_ptr<TensorList>& out_grads, bool retain_graph, bool create_graph) {
  auto capture_tensors = std::make_shared<TensorList>(inputs->size());
  bool retain_grad_for_leaf = inputs->empty();
  for (int i = 0; i < outputs->size(); i++) {
    outputs->at(i)->now_grad.lock()->set_tensor_ptr(out_grads->at(i));
  }
  auto it = node_list_.begin();
  while (it != node_list_.end()) {
    if (it->lock()) {
      it->lock()->Apply(create_graph);

      // TODO: capture return grads and save grad for tensors whose retain_grad is true

      if (!retain_graph) {
        node_list_.erase(it);
      } else {
        it++;
      }
    } else {
      node_list_.erase(it);
    }
  }
  return capture_tensors;
}

}  // namespace one

}  // namespace oneflow
