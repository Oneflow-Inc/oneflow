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

#include <list>
#include <vector>
#include <memory>
#include <functional>
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace one {

class TensorArg;
class Tensor;
class TensorList;

// Calculates one backward op
class FunctionNode {
 public:
  virtual ~FunctionNode() = default;

  virtual void Apply(bool create_graph) = 0;
  virtual void ReleaseOutTensorArgs() = 0;
  // Releases the eventual c++ std::function for backward if retain_graph=False to avoid calling
  // `Apply` in second time
  virtual void ReleaseGraph() = 0;

  // Getters
  virtual std::shared_ptr<std::vector<std::shared_ptr<const FunctionNode>>> GetNextFunctions() = 0;
  virtual const std::string& GetOpName() const = 0;

 protected:
  FunctionNode() = default;
};

class AutogradEngine {
 public:
  virtual ~AutogradEngine() = default;

  // Calls every `FunctionNode.Apply()` and capture grad in this calling for `inputs`
  virtual Maybe<std::shared_ptr<TensorList>> Execute(const TensorList& outputs,
                                                     const TensorList& inputs,
                                                     const TensorList& out_grads, bool retain_graph,
                                                     bool create_graph) = 0;
  // Builds FunctionNode, binding to all `outputs_` tensors and saving in AutogradEngine
  // TODO: add parameters for `backward_fn`
  virtual const std::shared_ptr<FunctionNode>& AddBackwardFuncPtr(
      const std::shared_ptr<const std::function<void()>>& backward_fn, const TensorList& inputs,
      const TensorList& outputs) = 0;

 protected:
  AutogradEngine() = default;
};

// Stack Autograd Node and Engine
class StackFunctionNode final : public FunctionNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StackFunctionNode);
  // TODO: update constructor according to op_builder interface
  StackFunctionNode(const std::shared_ptr<const std::function<void()>>& backward_fn,
                    const TensorList& inputs, const TensorList& outputs);
  ~StackFunctionNode() override = default;

  std::shared_ptr<std::vector<std::shared_ptr<const FunctionNode>>> GetNextFunctions() override {
    return next_functions_;
  }
  const std::string& GetOpName() const override { return op_name_; }

  void ReleaseOutTensorArgs() override;
  void ReleaseGraph() override;
  void Apply(bool create_graph) override;

 private:
  // FunctionNode shares Tensor with `inputs_`, and only shares TensorImpl with `outputs_`.
  // The reference link is `output tensors -> node -> inputs_/input tensors`.
  std::shared_ptr<TensorList> inputs_;
  std::shared_ptr<TensorList> outputs_;
  std::vector<std::shared_ptr<TensorArg>> in_grads_;
  std::vector<std::shared_ptr<TensorArg>> out_grads_;
  // Actual backward function builds in `AutogradInterpreter` to calculate one backward op
  // TODO: add parameters
  std::shared_ptr<const std::function<void()>> backward_fn_;

  const std::string op_name_;
  std::shared_ptr<std::vector<std::shared_ptr<const FunctionNode>>> next_functions_;
};

class StackAutogradEngine final : public AutogradEngine {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StackAutogradEngine);
  StackAutogradEngine() = default;
  ~StackAutogradEngine() override = default;

  Maybe<std::shared_ptr<TensorList>> Execute(const TensorList& outputs, const TensorList& inputs,
                                             const TensorList& out_grads, bool retain_graph,
                                             bool create_graph) override;
  const std::shared_ptr<FunctionNode>& AddBackwardFuncPtr(
      const std::shared_ptr<const std::function<void()>>& backward_fn, const TensorList& inputs,
      const TensorList& outputs) override;

 protected:
  // StackFunctionNode must be saved in engine, because any node in list may be released at any
  // moment.
  std::list<std::weak_ptr<StackFunctionNode>> node_list_;
};

}  // namespace one

}  // namespace oneflow
#endif  // ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_ENGINE_H_
