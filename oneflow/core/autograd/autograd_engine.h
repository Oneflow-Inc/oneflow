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
  std::shared_ptr<TensorList> inputs_;
  std::shared_ptr<TensorList> outputs_;
  std::vector<std::shared_ptr<TensorArg>> in_grads_;
  std::vector<std::shared_ptr<TensorArg>> out_grads_;
  // TODO: add parameters
  std::function<void()> backward_fn_;
};

class AutogradEngine {
 public:
  AutogradEngine() = default;
  virtual ~AutogradEngine() = default;

  virtual std::shared_ptr<TensorList> Execute(const std::shared_ptr<TensorList>& outputs,
                                              const std::shared_ptr<TensorList>& inputs,
                                              const std::shared_ptr<TensorList>& out_grads,
                                              bool retain_graph, bool create_graph) = 0;
  // TODO: add parameters
  virtual std::shared_ptr<FunctionNode> AddBackwardFuncPtr(
      std::function<void()>, const std::shared_ptr<TensorList>& inputs,
      const std::shared_ptr<TensorList>& outputs) = 0;
};

// Stack AutogradEngine
class StackFunctionNode final : public FunctionNode {
 public:
  StackFunctionNode(std::function<void()> backward_fn, const std::shared_ptr<TensorList>& intputs,
                    const std::shared_ptr<TensorList>& outputs);
  ~StackFunctionNode() = default;

  std::weak_ptr<StackFunctionNode> GetPrevNode() { return prev_node_; }
  void Release() override;
  void Apply(bool create_graph) override;

 protected:
  std::weak_ptr<StackFunctionNode> prev_node_;
};

class StackAutogradEngine final : public AutogradEngine {
 public:
  std::shared_ptr<TensorList> Execute(const std::shared_ptr<TensorList>& outputs,
                                      const std::shared_ptr<TensorList>& inputs,
                                      const std::shared_ptr<TensorList>& out_grads,
                                      bool retain_graph, bool create_graph) override;
  virtual std::shared_ptr<FunctionNode> AddBackwardFuncPtr(
      std::function<void()>, const std::shared_ptr<TensorList>& inputs,
      const std::shared_ptr<TensorList>& outputs) override;

 protected:
  std::weak_ptr<StackFunctionNode> top_node_;
};

}  // namespace one

}  // namespace oneflow

