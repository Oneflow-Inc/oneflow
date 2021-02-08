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

std::shared_ptr<TensorList> MakeGrads(std::shared_ptr<TensorList> outputs,
                                      std::shared_ptr<TensorList> out_grads) {
  std::shared_ptr<TensorList> gradients;
  // TODO: check all out_grads and push default value for empty item
  return gradients;
}

std::shared_ptr<TensorList> RunBackward(std::shared_ptr<TensorList> outputs,
                                        std::shared_ptr<TensorList> intputs,
                                        std::shared_ptr<TensorList> out_grads, bool retain_graph,
                                        bool create_graph) {
  if (create_graph) retain_graph = true;
  std::shared_ptr<TensorList> res_grads;
  // TODO: check could run backward or not
  // TODO: add backward codes
  return res_grads;
}

}  // namespace

namespace one {

std::shared_ptr<TensorList> Backward(std::shared_ptr<TensorList> outputs,
                                     std::shared_ptr<TensorList> out_grads, bool retain_graph,
                                     bool create_graph) {
  std::shared_ptr<TensorList> gradients = MakeGrads(outputs, out_grads);
  std::shared_ptr<TensorList> res_grads =
      RunBackward(outputs, {}, gradients, retain_graph, create_graph);
  // TODO: bind res_grads to all inputs
  return {};
}

std::shared_ptr<TensorList> Grad(std::shared_ptr<TensorList> outputs,
                                 std::shared_ptr<TensorList> inputs,
                                 std::shared_ptr<TensorList> out_grads, bool retain_graph,
                                 bool create_graph) {
  if (inputs->empty()) return Backward(outputs, out_grads, retain_graph, create_graph);

  std::shared_ptr<TensorList> gradients = MakeGrads(outputs, out_grads);
  return RunBackward(outputs, inputs, gradients, retain_graph, create_graph);
}

}  // namespace one

}  // namespace oneflow
