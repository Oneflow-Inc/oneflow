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

#ifndef ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_ENGINE_H_
#define ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_ENGINE_H_

#include <vector>
#include <memory>
#include <functional>
#include <list>

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
  virtual void ReleaseTensorGrads() = 0;
  // Release backward_fn_ if retain_graph=False to avoid call Apply in second time
  virtual void ReleaseGraph() = 0;

  // Getters
  virtual std::vector<std::weak_ptr<FunctionNode>> GetNextFunctions() = 0;
  virtual const std::string& GetOpName() = 0;
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
  // TODO: update constructor according to op_builder interface
  StackFunctionNode(std::function<void()> backward_fn, const std::shared_ptr<TensorList>& inputs,
                    const std::shared_ptr<TensorList>& outputs);

  std::vector<std::weak_ptr<FunctionNode>> GetNextFunctions() override { return next_functions_; }
  const std::string& GetOpName() override { return op_name_; }

  void ReleaseTensorGrads() override;
  void ReleaseGraph() override;
  void Apply(bool create_graph) override;

 private:
  /*
   * FunctionNode shares tensor_impl with `inputs_`, and only shares blob pointer with `outputs_`.
   * The reference link is `output tensors -> node -> inputs_/input tensors`.
   */
  std::shared_ptr<TensorList> inputs_;
  std::shared_ptr<TensorList> outputs_;
  std::vector<std::shared_ptr<TensorArg>> in_grads_;
  std::vector<std::shared_ptr<TensorArg>> out_grads_;
  // TODO: add parameters
  std::shared_ptr<std::function<void()>> backward_fn_;

  const std::string op_name_;
  std::vector<std::weak_ptr<FunctionNode>> next_functions_;
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
  /*
   * StackFunctionNode must be saved in engine, because any node in list may be released at any
   * moment.
   */
  std::list<std::weak_ptr<StackFunctionNode>> node_list_;
};

}  // namespace one

}  // namespace oneflow
#endif  // ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_ENGINE_H_
