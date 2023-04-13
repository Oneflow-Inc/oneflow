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

#include <functional>
#include <list>
#include <memory>
#include <vector>

#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {

namespace one {

class Tensor;
class TensorTuple;

using CaptureStatus = bool;

struct BackwardFunction {
  std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)> body;
  std::function<CaptureStatus()> status;
};

// Calculates one backward op
class FunctionNode {
 public:
  virtual ~FunctionNode() = default;

  Maybe<bool> Apply(bool create_graph);
  Maybe<void> AccGrad4LeafTensor(bool create_graph);
  Maybe<void> AccGrad4RetainGradTensor(bool create_graph);
  void ReleaseOutTensorArgs();
  // Releases the eventual c++ std::function for backward if retain_graph=False to avoid calling
  // `Apply` in second time
  virtual void ReleaseData() = 0;

  const std::vector<std::shared_ptr<FunctionNode>>& next_functions() const {
    return next_functions_;
  }
  const std::string& name() const { return name_; }

  const std::shared_ptr<Scope>& scope() const { return scope_; }
  void set_scope(const std::shared_ptr<Scope>& scope) { scope_ = scope; }

  using Hook = std::function<Optional<std::vector<std::shared_ptr<Tensor>>>(const TensorTuple&,
                                                                            const TensorTuple&)>;
  void add_post_hook(const Hook& hook) { hooks_.push_back(hook); }

 protected:
  friend class GraphTask;
  explicit FunctionNode(const std::string& name,
                        const std::shared_ptr<BackwardFunction>& backward_fn)
      : name_(name), backward_fn_(backward_fn), scope_(nullptr) {}

  const std::string name_;
  std::vector<std::shared_ptr<FunctionNode>> next_functions_;

  std::vector<std::shared_ptr<AutogradMeta>> input_meta_data_;
  std::vector<std::shared_ptr<AutogradMeta>> output_meta_data_;
  std::vector<TensorInfo> output_tensor_infos_;

  // Actual backward function builds in `AutogradInterpreter` to calculate one backward op
  std::shared_ptr<BackwardFunction> backward_fn_;

  // The execution scope
  std::shared_ptr<Scope> scope_;

  std::vector<Hook> hooks_;
};

class AutogradEngine {
 public:
  virtual ~AutogradEngine() = default;

  Maybe<void> RunBackwardAndSaveGrads4LeafTensorIf(const TensorTuple& outputs,
                                                   const TensorTuple& out_grads, bool retain_graph,
                                                   bool create_graph);
  Maybe<TensorTuple> RunBackwardAndReturnInputsTensorGradIf(const TensorTuple& outputs,
                                                            const TensorTuple& inputs,
                                                            const TensorTuple& out_grads,
                                                            bool retain_graph, bool create_graph);
  virtual void ClearEngine() = 0;
  // Builds FunctionNode, binding to all `outputs_` tensors and saving in AutogradEngine
  virtual Maybe<FunctionNode> AddNode(const std::string& name,
                                      const std::shared_ptr<BackwardFunction>& backward_fn,
                                      const TensorTuple& inputs, TensorTuple* outputs) = 0;

 protected:
  AutogradEngine() = default;

 private:
  virtual Maybe<void> RunBackwardAndSaveGrads4LeafTensor(const TensorTuple& outputs,
                                                         const TensorTuple& out_grads,
                                                         bool retain_graph, bool create_graph) = 0;
  virtual Maybe<TensorTuple> RunBackwardAndReturnInputsTensorGrad(const TensorTuple& outputs,
                                                                  const TensorTuple& inputs,
                                                                  const TensorTuple& out_grads,
                                                                  bool retain_graph,
                                                                  bool create_graph) = 0;
};

// Graph Autograd Node and Engine
class GraphFunctionNode final : public FunctionNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GraphFunctionNode);
  static std::shared_ptr<GraphFunctionNode> New(
      const std::string& name, const std::shared_ptr<BackwardFunction>& backward_fn,
      const TensorTuple& inputs, const TensorTuple& outputs);

  GraphFunctionNode() = delete;
  ~GraphFunctionNode() override = default;

  void ReleaseData() override;

 private:
  GraphFunctionNode(const std::string& name, const std::shared_ptr<BackwardFunction>& backward_fn,
                    const TensorTuple& inputs, const TensorTuple& outputs);
};

class GraphTask final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GraphTask);
  GraphTask() = delete;
  GraphTask(const TensorTuple& outputs, bool retain_graph, bool create_graph);

  Maybe<void> ComputeDependencies();
  Maybe<void> ComputeDependenciesAndPruneNode(const TensorTuple& inputs);
  Maybe<void> Apply(bool save_grad_for_leaf);
  std::shared_ptr<TensorTuple> GetCapturedGrads() const { return captured_grads_; }
  Maybe<void> WriteGraphToDotFile(const std::string& file_name) const;

 private:
  class ExecInfo {
   public:
    ExecInfo() = default;

    int32_t dependencies = 0;
    bool need_execute = false;
    // Used in autograd.grad interface, to record which grad of tensor will be captured.
    // The pair means: <output index of this Node, the index of captured_grads_ to be saved>
    std::unique_ptr<std::vector<std::pair<size_t, size_t>>> capture_indices;
  };

  bool retain_graph_;
  bool create_graph_;
  std::vector<FunctionNode*> roots_;
  HashMap<FunctionNode*, ExecInfo> grad_fn2exec_info_;
  std::shared_ptr<TensorTuple> captured_grads_;
};

class GraphAutogradEngine final : public AutogradEngine {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GraphAutogradEngine);
  GraphAutogradEngine() = default;
  ~GraphAutogradEngine() override = default;

  void ClearEngine() override{};
  Maybe<FunctionNode> AddNode(const std::string& name,
                              const std::shared_ptr<BackwardFunction>& backward_fn,
                              const TensorTuple& inputs, TensorTuple* outputs) override;

 private:
  Maybe<void> RunBackwardAndSaveGrads4LeafTensor(const TensorTuple& outputs,
                                                 const TensorTuple& out_grads, bool retain_graph,
                                                 bool create_graph) override;
  Maybe<TensorTuple> RunBackwardAndReturnInputsTensorGrad(const TensorTuple& outputs,
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
