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

#include <pybind11/pybind11.h>
#include <memory>
#include <vector>
#include "oneflow/api/python/autograd/autograd.h"
#include "oneflow/api/python/autograd/autograd_api.h"
#include "oneflow/api/python/of_api_registry.h"

namespace oneflow {
namespace autograd {

namespace {

// Checks and sets default value for initial gradients based on out_grads
// If output is the tensor whose size is greater than 1, out_grad's shape must be same as output's.
// If output is a scaler tensor, out_grad will also be a scaler or empty(will be inited to
// `flow.ones([1])`).
std::shared_ptr<one::TensorList> CheckAndInitOutGrads(one::TensorList* outputs,
                                                      one::TensorList* out_grads) {
  auto gradients = std::make_shared<one::TensorList>(out_grads->size());
  // TODO: check all out_grads and push default value for empty item
  return gradients;
}

// All autograd operators will call this function finally to calculate gradients for each input by
// calling once `AutogradEngine.Execute()`
std::shared_ptr<one::TensorList> RunBackward(one::TensorList* outputs, one::TensorList* intputs,
                                             one::TensorList* out_grads, bool retain_graph,
                                             bool create_graph) {
  if (create_graph) { retain_graph = true; }
  std::shared_ptr<one::TensorList> res_grads;
  // TODO: check could run backward or not
  // TODO: calls `AutogradEngine.Execute()` to do backward
  return res_grads;
}

}  // namespace

Maybe<std::shared_ptr<one::TensorList>> Backward(const std::shared_ptr<one::TensorList>& outputs,
                                                 const std::shared_ptr<one::TensorList>& out_grads,
                                                 bool retain_graph, bool create_graph) {
  std::shared_ptr<one::TensorList> gradients = CheckAndInitOutGrads(outputs.get(), out_grads.get());
  auto inputs = std::make_shared<one::TensorList>(0);
  return RunBackward(outputs.get(), inputs.get(), gradients.get(), retain_graph, create_graph);
}

Maybe<std::shared_ptr<one::TensorList>> Grad(const std::shared_ptr<one::TensorList>& outputs,
                                             const std::shared_ptr<one::TensorList>& inputs,
                                             const std::shared_ptr<one::TensorList>& out_grads,
                                             bool retain_graph, bool create_graph) {
  if (inputs->empty()) { return Backward(outputs, out_grads, retain_graph, create_graph); }

  std::shared_ptr<one::TensorList> gradients = CheckAndInitOutGrads(outputs.get(), out_grads.get());
  return RunBackward(outputs.get(), inputs.get(), gradients.get(), retain_graph, create_graph);
}

}  // namespace autograd
}  // namespace oneflow

ONEFLOW_API_PYBIND11_MODULE("autograd", m) {
  m.def("backward", &Backward);
  m.def("grad", &Grad);
}
