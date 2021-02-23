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

// check and set default value for initial gradients based on out_grads
std::shared_ptr<TensorList> MakeGrads(const std::shared_ptr<TensorList>& outputs,
                                      const std::shared_ptr<TensorList>& out_grads) {
  auto gradients = std::make_shared<TensorList>(out_grads->size());
  // TODO: check all out_grads and push default value for empty item
  return gradients;
}

// call AutogradEngine.Execute() to calculate gradients
std::shared_ptr<TensorList> RunBackward(const std::shared_ptr<TensorList>& outputs,
                                        const std::shared_ptr<TensorList>& intputs,
                                        const std::shared_ptr<TensorList>& out_grads,
                                        bool retain_graph, bool create_graph) {
  if (create_graph) retain_graph = true;
  std::shared_ptr<TensorList> res_grads;
  // TODO: check could run backward or not
  // TODO: add backward codes
  return res_grads;
}

}  // namespace

namespace one {

// export to python as autograd.backward()
std::shared_ptr<TensorList> Backward(const std::shared_ptr<TensorList>& outputs,
                                     const std::shared_ptr<TensorList>& out_grads,
                                     bool retain_graph, bool create_graph) {
  std::shared_ptr<TensorList> gradients = MakeGrads(outputs, out_grads);
  auto inputs = std::make_shared<TensorList>(0);
  std::shared_ptr<TensorList> res_grads =
      RunBackward(outputs, inputs, gradients, retain_graph, create_graph);
  return std::make_shared<TensorList>(0);
}

// export to python as autograd.grad()
std::shared_ptr<TensorList> Grad(const std::shared_ptr<TensorList>& outputs,
                                 const std::shared_ptr<TensorList>& inputs,
                                 const std::shared_ptr<TensorList>& out_grads,
                                 bool retain_graph, bool create_graph) {
  if (inputs->empty()) {
      return Backward(outputs, out_grads, retain_graph, create_graph);
  }

  std::shared_ptr<TensorList> gradients = MakeGrads(outputs, out_grads);
  return RunBackward(outputs, inputs, gradients, retain_graph, create_graph);
}

}  // namespace one

}  // namespace oneflow
