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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace autograd {

namespace {

// Checks and sets default value for initial gradients based on out_grads
// If output is the tensor whose size is greater than 1, out_grad's shape must be same as output's.
// If output is a scaler tensor, out_grad will also be a scaler or empty(will be inited to
// `flow.ones([1])`).
Maybe<one::TensorTuple> CheckAndInitOutGrads(const one::TensorTuple& outputs,
                                             const one::TensorTuple& out_grads) {
  auto gradients = std::make_shared<one::TensorTuple>(out_grads.size());
  TODO();  // Wang Yinggang. Checks all out_grads and push default value for empty item
  return gradients;
}

}  // namespace

Maybe<one::TensorTuple> Backward(const one::TensorTuple& outputs, const one::TensorTuple& out_grads,
                                 bool retain_graph, bool create_graph) {
  if (create_graph) { retain_graph = true; }
  std::shared_ptr<one::TensorTuple> gradients = JUST(CheckAndInitOutGrads(outputs, out_grads));
  JUST(one::GetThreadLocalAutogradEngine()->RunBackwardAndSaveGrads4LeafTensor(
      outputs, *gradients, retain_graph, create_graph));
  return std::make_shared<one::TensorTuple>(0);
}

Maybe<one::TensorTuple> Grad(const one::TensorTuple& outputs, const one::TensorTuple& inputs,
                             const one::TensorTuple& out_grads, bool retain_graph,
                             bool create_graph) {
  if (create_graph) { retain_graph = true; }
  if (inputs.empty()) { return Backward(outputs, out_grads, retain_graph, create_graph); }
  std::shared_ptr<one::TensorTuple> gradients = JUST(CheckAndInitOutGrads(outputs, out_grads));
  return one::GetThreadLocalAutogradEngine()->RunBackwardAndReturnInputsTensorGrad(
      outputs, inputs, *gradients, retain_graph, create_graph);
}

}  // namespace autograd

namespace {

std::shared_ptr<oneflow::one::TensorTuple> BackwardOrThrow(
    const std::shared_ptr<oneflow::one::TensorTuple>& outputs,
    const std::shared_ptr<oneflow::one::TensorTuple>& out_grads, bool retain_graph,
    bool create_graph) {
  return oneflow::autograd::Backward(*outputs, *out_grads.get(), retain_graph, create_graph)
      .GetPtrOrThrow();
}

std::shared_ptr<oneflow::one::TensorTuple> GradOrThrow(
    const std::shared_ptr<oneflow::one::TensorTuple>& outputs,
    const std::shared_ptr<oneflow::one::TensorTuple>& inputs,
    const std::shared_ptr<oneflow::one::TensorTuple>& out_grads, bool retain_graph,
    bool create_graph) {
  return oneflow::autograd::Grad(*outputs, *inputs, *out_grads.get(), retain_graph, create_graph)
      .GetPtrOrThrow();
}

ONEFLOW_API_PYBIND11_MODULE("autograd", m) {
  m.def("backward", &BackwardOrThrow);
  m.def("grad", &GradOrThrow);
}

}  // namespace

}  // namespace oneflow
