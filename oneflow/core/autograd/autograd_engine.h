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
#include <vector>
#include <memory>
#include <functional>

namespace oneflow {

namespace one {

class TensorArg;
class Tensor;
using TensorList = std::vector<std::shared_ptr<Tensor>>;

class FunctionNode {
 public:
  FunctionNode() = default;
  virtual ~FunctionNode() = default;

  virtual void Apply(bool create_graph) = 0;
  virtual void Release() = 0;

 protected:
  TensorList inputs;
  TensorList outputs;
  std::vector<std::shared_ptr<TensorArg>> in_grads;
  std::vector<std::shared_ptr<TensorArg>> out_grads;
  // TODO: add parameters
  std::function<void()> backward_fn_;
};

class AutogradEngine {
 public:
  AutogradEngine() = default;
  virtual ~AutogradEngine() = default;

  virtual std::shared_ptr<TensorList> Execute(std::shared_ptr<TensorList> outputs,
                                              std::shared_ptr<TensorList> inputs,
                                              std::shared_ptr<TensorList> out_grads,
                                              bool retain_graph, bool create_graph) = 0;
  // TODO: add parameters
  virtual std::shared_ptr<FunctionNode> AddBackwardFuncPtr(std::function<void()>) = 0;
};

}  // namespace one

}  // namespace oneflow
