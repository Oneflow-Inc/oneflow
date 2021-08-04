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
#include "oneflow/core/autograd/autograd_meta.h"

namespace oneflow {

namespace one {

class Tensor;
class TensorTuple;

// Calculates one backward op
class FunctionNode {
 public:
  virtual ~FunctionNode() = default;

  Maybe<bool> Apply(bool create_graph);
  Maybe<void> AccGrad4LeafTensor(bool create_graph);
  Maybe<void> AccGrad4RetainGradTensor();
  void ReleaseOutTensorArgs();
  // Releases the eventual c++ std::function for backward if retain_graph=False to avoid calling
  // `Apply` in second time
  virtual void ReleaseData() = 0;

  // Getters
  const std::shared_ptr<std::vector<std::shared_ptr<FunctionNode>>>& GetNextFunctions() const {
    return next_functions_;
  }
  const std::string& GetOpTypeName() const { return op_name_; }

 protected:
  explicit FunctionNode(const std::string& op_type_name)
      : op_name_(op_type_name), next_functions_(new std::vector<std::shared_ptr<FunctionNode>>{}) {}

  const std::string op_name_;
  std::shared_ptr<std::vector<std::shared_ptr<FunctionNode>>> next_functions_;

  std::vector<std::shared_ptr<AutogradMeta>> input_meta_datas_;
  std::vector<std::shared_ptr<AutogradMeta>> output_meta_datas_;
  std::vector<TensorInfo> output_tensor_infos_;
  // Actual backward function builds in `AutogradInterpreter` to calculate one backward op
  std::shared_ptr<const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>
      backward_fn_;
};

class AutogradEngine {
 public:
  virtual ~AutogradEngine() = default;

  Maybe<void> RunBackwardAndSaveGrads4LeafTensor(const TensorTuple& outputs,
                                                 const TensorTuple& out_grads, bool retain_graph,
                                                 bool create_graph);
  Maybe<TensorTuple> RunBackwardAndReturnInputsTensorGrad(const TensorTuple& outputs,
                                                          const TensorTuple& inputs,
                                                          const TensorTuple& out_grads,
                                                          bool retain_graph, bool create_graph);
  virtual void ClearEngine() = 0;
  // Builds FunctionNode, binding to all `outputs_` tensors and saving in AutogradEngine
  virtual Maybe<FunctionNode> AddBackwardFuncPtr(
      const std::string& op_type_name,
      const std::shared_ptr<
          const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>& backward_fn,
      const TensorTuple& inputs, TensorTuple* outputs) = 0;

 protected:
  AutogradEngine() = default;

 private:
  virtual Maybe<void> RunBackwardAndSaveGrads4LeafTensorIf(const TensorTuple& outputs,
                                                           const TensorTuple& out_grads,
                                                           bool retain_graph,
                                                           bool create_graph) = 0;
  virtual Maybe<TensorTuple> RunBackwardAndReturnInputsTensorGradIf(const TensorTuple& outputs,
                                                                    const TensorTuple& inputs,
                                                                    const TensorTuple& out_grads,
                                                                    bool retain_graph,
                                                                    bool create_graph) = 0;
};

// Stack Autograd Node and Engine
class StackFunctionNode final : public FunctionNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StackFunctionNode);
  StackFunctionNode(
      const std::string& op_type_name,
      const std::shared_ptr<
          const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>& backward_fn,
      const TensorTuple& inputs, const TensorTuple& outputs);
  StackFunctionNode() = delete;
  ~StackFunctionNode() override = default;

  void ReleaseData() override;
  bool is_in_stack() const { return is_in_stack_; }
  void set_is_in_stack(bool in_stack) { is_in_stack_ = in_stack; }

 private:
  bool is_in_stack_;
};

class StackAutogradEngine final : public AutogradEngine {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StackAutogradEngine);
  StackAutogradEngine() = default;
  ~StackAutogradEngine() override = default;

  void ClearEngine() override;
  Maybe<FunctionNode> AddBackwardFuncPtr(
      const std::string& op_type_name,
      const std::shared_ptr<
          const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>& backward_fn,
      const TensorTuple& inputs, TensorTuple* outputs) override;

 private:
  // StackFunctionNode must be saved in engine, because any node in list may be released at any
  // moment.
  std::list<std::weak_ptr<FunctionNode>> node_list_;
  void ClearReleasedFunctionNodes();
  Maybe<void> RunBackwardAndSaveGrads4LeafTensorIf(const TensorTuple& outputs,
                                                   const TensorTuple& out_grads, bool retain_graph,
                                                   bool create_graph) override;
  Maybe<TensorTuple> RunBackwardAndReturnInputsTensorGradIf(const TensorTuple& outputs,
                                                            const TensorTuple& inputs,
                                                            const TensorTuple& out_grads,
                                                            bool retain_graph,
                                                            bool create_graph) override;
};

// Graph Autograd Node and Engine
class GraphFunctionNode final : public FunctionNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GraphFunctionNode);
  GraphFunctionNode(
      const std::string& op_type_name,
      const std::shared_ptr<
          const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>& backward_fn,
      const TensorTuple& inputs, const TensorTuple& outputs);
  GraphFunctionNode() = delete;
  ~GraphFunctionNode() override = default;

  void ReleaseData() override;
};

class GraphTask final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GraphTask);
  GraphTask() = delete;
  GraphTask(const TensorTuple& outputs, bool retain_graph, bool create_graph);

  Maybe<void> ComputeDependencies();
  Maybe<void> ComputeDependenciesAndPruneNode(const TensorTuple& inputs);
  Maybe<void> Apply(bool save_grad_for_leaf);

 private:
  bool retain_graph_;
  bool create_graph_;
  std::vector<FunctionNode*> roots_;
  HashMap<FunctionNode*, int> dependencies_;
  HashSet<FunctionNode*> need_execute_;
};

class GraphAutogradEngine final : public AutogradEngine {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GraphAutogradEngine);
  GraphAutogradEngine() = default;
  ~GraphAutogradEngine() override = default;

  void ClearEngine() override{};
  Maybe<FunctionNode> AddBackwardFuncPtr(
      const std::string& op_type_name,
      const std::shared_ptr<
          const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>& backward_fn,
      const TensorTuple& inputs, TensorTuple* outputs) override;

 private:
  Maybe<void> RunBackwardAndSaveGrads4LeafTensorIf(const TensorTuple& outputs,
                                                   const TensorTuple& out_grads, bool retain_graph,
                                                   bool create_graph) override;
  Maybe<TensorTuple> RunBackwardAndReturnInputsTensorGradIf(const TensorTuple& outputs,
                                                            const TensorTuple& inputs,
                                                            const TensorTuple& out_grads,
                                                            bool retain_graph,
                                                            bool create_graph) override;
};

AutogradEngine* GetThreadLocalAutogradEngine();

Maybe<void> AddAccumulateFunctionNode(const std::shared_ptr<Tensor>& tensor);

}  // namespace one

}  // namespace oneflow
#endif  // ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_ENGINE_H_
