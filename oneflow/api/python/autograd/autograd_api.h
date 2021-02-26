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

#ifndef ONEFLOW_API_PYTHON_AUTOGRAD_AUTOGRAD_API_H
#define ONEFLOW_API_PYTHON_AUTOGRAD_AUTOGRAD_API_H

#include <memory>
#include <vector>
#include "oneflow/api/python/autograd/autograd.h"
#include "oneflow/core/framework/tensor_list.h"

inline std::shared_ptr<oneflow::one::TensorList> Backward(
    const std::shared_ptr<oneflow::one::TensorList>& outputs,
    const std::shared_ptr<oneflow::one::TensorList>& out_grads, bool retain_graph,
    bool create_graph) {
  return oneflow::autograd::Backward(outputs, out_grads, retain_graph, create_graph).GetOrThrow();
}

inline std::shared_ptr<oneflow::one::TensorList> Grad(
    const std::shared_ptr<oneflow::one::TensorList>& outputs,
    const std::shared_ptr<oneflow::one::TensorList>& inputs,
    const std::shared_ptr<oneflow::one::TensorList>& out_grads, bool retain_graph,
    bool create_graph) {
  return oneflow::autograd::Grad(outputs, inputs, out_grads, retain_graph, create_graph)
      .GetOrThrow();
}

#endif
