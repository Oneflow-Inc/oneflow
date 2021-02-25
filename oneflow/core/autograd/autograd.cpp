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

#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

// Checks and sets default value for initial gradients based on out_grads
// If output is the tensor whose size is greater than 1, out_grad's shape must be same as output's.
// If output is a scaler tensor, out_grad will also be a scaler or empty(will be inited to
// `flow.ones([1])`).
std::shared_ptr<TensorList> CheckAndInitOutGrads(const std::shared_ptr<TensorList>& outputs,
                                                 const std::shared_ptr<TensorList>& out_grads) {
  auto gradients = std::make_shared<TensorList>(out_grads->size());
  // TODO: check all out_grads and push default value for empty item
  return gradients;
}

// Calculates gradients for each input by launching AutogradEngine
std::shared_ptr<TensorList> RunBackward(const std::shared_ptr<TensorList>& outputs,
                                        const std::shared_ptr<TensorList>& intputs,
                                        const std::shared_ptr<TensorList>& out_grads,
                                        bool retain_graph, bool create_graph) {
  if (create_graph) { retain_graph = true; }
  std::shared_ptr<TensorList> res_grads;
  // TODO: check could run backward or not
  // TODO: calls `AutogradEngine.Execute()` to do backward
  return res_grads;
}

}  // namespace

std::shared_ptr<TensorList> AutoGradUtil::Backward(const std::shared_ptr<TensorList>& outputs,
                                                   const std::shared_ptr<TensorList>& out_grads,
                                                   bool retain_graph, bool create_graph) {
  std::shared_ptr<TensorList> gradients = CheckAndInitOutGrads(outputs, out_grads);
  auto inputs = std::make_shared<TensorList>(0);
  return RunBackward(outputs, inputs, gradients, retain_graph, create_graph);
}

std::shared_ptr<TensorList> AutoGradUtil::Grad(const std::shared_ptr<TensorList>& outputs,
                                               const std::shared_ptr<TensorList>& inputs,
                                               const std::shared_ptr<TensorList>& out_grads,
                                               bool retain_graph, bool create_graph) {
  if (inputs->empty()) { return Backward(outputs, out_grads, retain_graph, create_graph); }

  std::shared_ptr<TensorList> gradients = CheckAndInitOutGrads(outputs, out_grads);
  return RunBackward(outputs, inputs, gradients, retain_graph, create_graph);

}  // namespace one

}  // namespace oneflow
