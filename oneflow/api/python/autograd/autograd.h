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

#ifndef ONEFLOW_API_PYTHON_AUTOGRAD_AUTOGRAD_H_
#define ONEFLOW_API_PYTHON_AUTOGRAD_AUTOGRAD_H_

#include <memory>
#include <vector>
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/tensor_list.h"

namespace oneflow {
namespace autograd {

Maybe<std::shared_ptr<one::TensorList>> Backward(const std::shared_ptr<one::TensorList>& outputs,
                                                 const std::shared_ptr<one::TensorList>& out_grads,
                                                 bool retain_graph, bool create_graph);

Maybe<std::shared_ptr<one::TensorList>> Grad(const std::shared_ptr<one::TensorList>& outputs,
                                             const std::shared_ptr<one::TensorList>& inputs,
                                             const std::shared_ptr<one::TensorList>& out_grads,
                                             bool retain_graph, bool create_graph);

}  // namespace autograd
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_AUTOGRAD_AUTOGRAD_H_
