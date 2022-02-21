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

#include <stack>
#include <queue>
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_arg.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/eager/dev_vm_dep_object_consume_mode.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

namespace {

bool IsReadyToRun(const std::vector<std::shared_ptr<AutogradMeta>>& out_meta_datas) {
  return std::any_of(out_meta_datas.begin(), out_meta_datas.end(),
                     [](const std::shared_ptr<AutogradMeta>& meta_data) {
                       return !meta_data->current_grad()->Empty();
                     });
}

Maybe<void> CopyOrAccGrad(AutogradMeta* autograd_meta, bool autograd_mode) {
  autograd::AutoGradMode mode(autograd_mode);
  auto current_grad = JUST(autograd_meta->current_grad()->GetAccTensor());
  if (!current_grad) { return Maybe<void>::Ok(); }
  for (const auto& hook : autograd_meta->hooks()) {
    auto new_grad = hook(current_grad);
    if (new_grad) { current_grad = new_grad; }
  }
  if (autograd_meta->acc_grad()) {
    DevVmDepObjectConsumeModeGuard guard(DevVmDepObjectConsumeMode::NONE);
    // Should not inplace accumulate grad. For example,
    // >>> z = x + y
    // >>> p = x / z
    // >>> p.sum().backward()
    //
    // As we know that dx = dz + dp / z and dy = dz, so it will lead to wrong value
    // for dy if dx is shared with dz.
    const auto& output = JUST(functional::Add(autograd_meta->acc_grad(), current_grad, /*alpha=*/1,
                                              /*inplace=*/autograd_meta->is_grad_acc_inplace()));
    JUST(autograd_meta->set_acc_grad(output));
  } else {
    JUST(autograd_meta->set_acc_grad(current_grad));
  }
  for (const auto& hook : autograd_meta->post_grad_accumulation_hooks()) {
    auto new_grad = hook(autograd_meta->acc_grad());
    if (new_grad) { JUST(autograd_meta->set_acc_grad(new_grad)); }
  }

  return Maybe<void>::Ok();
}

Maybe<void> RawTorchConsistentTensor(const std::shared_ptr<one::Tensor>& tensor) {
  // Do nothing.
  return Maybe<void>::Ok();
}

static constexpr auto* TorchConsistentTensor =
    DECORATE(&RawTorchConsistentTensor, CheckConsistentTensorMeta);

Maybe<void> CheckConsistentTensorsMeta(const TensorTuple& tensor_tuple) {
  for (const auto& tensor : tensor_tuple) {
    if (tensor->is_consistent()) { JUST(TorchConsistentTensor(tensor)); }
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> AutogradEngine::RunBackwardAndSaveGrads4LeafTensorIf(const TensorTuple& outputs,
                                                                 const TensorTuple& out_grads,
                                                                 bool retain_graph,
                                                                 bool create_graph) {
  JUST(CheckConsistentTensorsMeta(outputs));
  JUST(CheckConsistentTensorsMeta(out_grads));
  DisableCheckConsistentTensorMetaScope disable_meta_check;
  return RunBackwardAndSaveGrads4LeafTensor(outputs, out_grads, retain_graph, create_graph);
}

Maybe<TensorTuple> AutogradEngine::RunBackwardAndReturnInputsTensorGradIf(
    const TensorTuple& outputs, const TensorTuple& inputs, const TensorTuple& out_grads,
    bool retain_graph, bool create_graph) {
  JUST(CheckConsistentTensorsMeta(outputs));
  JUST(CheckConsistentTensorsMeta(inputs));
  JUST(CheckConsistentTensorsMeta(out_grads));
  DisableCheckConsistentTensorMetaScope disable_meta_check;
  return RunBackwardAndReturnInputsTensorGrad(outputs, inputs, out_grads, retain_graph,
                                              create_graph);
}

StackFunctionNode::StackFunctionNode(
    const std::string& op_type_name,
    const std::shared_ptr<const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>&
        backward_fn,
    const TensorTuple& inputs, const TensorTuple& outputs)
    : FunctionNode(op_type_name) {
  input_meta_data_.resize(inputs.size());
  next_functions_->reserve(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    if (inputs.at(i)->requires_grad()) {
      input_meta_data_.at(i) = inputs.at(i)->mut_autograd_meta();
      next_functions_->emplace_back(inputs.at(i)->mut_grad_fn_node());
    }
  }

  output_meta_data_.resize(outputs.size());
  output_tensor_infos_.reserve(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    const auto& autograd_meta =
        NewAutogradMeta(outputs.at(i)->requires_grad(), outputs.at(i)->is_leaf());
    outputs.at(i)->set_autograd_meta(autograd_meta);
    output_meta_data_.at(i) = outputs.at(i)->mut_autograd_meta();
    output_tensor_infos_.emplace_back(TensorInfo(*outputs.at(i)));
  }

  backward_fn_ = backward_fn;
  is_in_stack_ = false;
}

Maybe<void> FunctionNode::AccGrad4RetainGradTensor() {
  for (const std::shared_ptr<AutogradMeta>& out : output_meta_data_) {
    if (out->retain_grad()) { JUST(CopyOrAccGrad(out.get(), /*autograd_mode=*/false)); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> FunctionNode::AccGrad4LeafTensor(bool create_graph) {
  for (const std::shared_ptr<AutogradMeta>& out : output_meta_data_) {
    if (out->is_leaf() && out->requires_grad()) {
      JUST(CopyOrAccGrad(out.get(), /*autograd_mode=*/false));
    }
  }
  return Maybe<void>::Ok();
}

void FunctionNode::ReleaseOutTensorArgs() {
  for (const std::shared_ptr<AutogradMeta>& meta_data : output_meta_data_) {
    meta_data->current_grad()->Release();
  }
}

void StackFunctionNode::ReleaseData() {
  if (!input_meta_data_.empty()) { backward_fn_.reset(); }
  is_in_stack_ = false;
}

Maybe<bool> FunctionNode::Apply(bool create_graph) {
  CHECK_NOTNULL_OR_RETURN(backward_fn_.get())
      << "This FunctionNode with name `" << GetOpTypeName() << "` has been released.\n"
      << "Maybe you try to backward through the node a second time. Specify retain_graph=True when "
         "calling .backward() or autograd.grad() the first time.";
  if (!IsReadyToRun(output_meta_data_)) { return false; }
  TensorTuple input_grads(input_meta_data_.size());
  TensorTuple output_grads(output_meta_data_.size());
  for (int i = 0; i < output_meta_data_.size(); ++i) {
    if (output_meta_data_.at(i)->current_grad()->Empty()) {
      output_grads.at(i) = JUST(output_tensor_infos_.at(i).zeros());
    } else {
      output_grads.at(i) = JUST(output_meta_data_.at(i)->current_grad()->GetAccTensor());
    }
  }
  JUST((*backward_fn_)(output_grads, &input_grads, create_graph));
  for (int i = 0; i < input_meta_data_.size(); ++i) {
    if (input_grads.at(i)) {
      CHECK_NOTNULL_OR_RETURN(input_meta_data_.at(i))
          << op_type_name_
          << " calculate grad for tensor which requires_grad is False. Please submit an issue in "
             "`https://github.com/Oneflow-Inc/oneflow/issues` and we will fix it as soon as "
             "possiable";
      JUST(input_meta_data_.at(i)->current_grad()->PushPartialTensor(input_grads.at(i)));
    }
  }
  return true;
}

void StackAutogradEngine::ClearEngine() { node_list_.clear(); }

void StackAutogradEngine::ClearReleasedFunctionNodes() {
  node_list_.erase(std::remove_if(node_list_.begin(), node_list_.end(),
                                  [](const std::weak_ptr<FunctionNode>& node) {
                                    return node.lock() == nullptr;
                                  }),
                   node_list_.end());
}

Maybe<void> StackAutogradEngine::RunBackwardAndSaveGrads4LeafTensor(const TensorTuple& outputs,
                                                                    const TensorTuple& out_grads,
                                                                    bool retain_graph,
                                                                    bool create_graph) {
  ClearReleasedFunctionNodes();
  for (int i = 0; i < outputs.size(); ++i) {
    JUST(JUST(outputs.at(i)->current_grad())->PushPartialTensor(out_grads.at(i)));
  }
  // Runs each FunctionNode
  for (const auto& weak_func_node : node_list_) {
    const auto& func_node = weak_func_node.lock();
    CHECK_NOTNULL_OR_RETURN(func_node);
    if (JUST(func_node->Apply(create_graph))) {
      JUST(func_node->AccGrad4LeafTensor(create_graph));
      JUST(func_node->AccGrad4RetainGradTensor());
      func_node->ReleaseOutTensorArgs();
      if (!retain_graph) { func_node->ReleaseData(); }
    }
  }
  if (!retain_graph) { ClearEngine(); }
  return Maybe<void>::Ok();
}

Maybe<TensorTuple> StackAutogradEngine::RunBackwardAndReturnInputsTensorGrad(
    const TensorTuple& outputs, const TensorTuple& inputs, const TensorTuple& out_grads,
    bool retain_graph, bool create_graph) {
  ClearReleasedFunctionNodes();
  std::shared_ptr<TensorTuple> input_current_grad = std::make_shared<TensorTuple>(inputs.size());
  std::vector<bool> ori_retain_grad(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    ori_retain_grad.at(i) = inputs.at(i)->retain_grad();
    JUST(inputs.at(i)->set_retain_grad(true));
  }
  for (int i = 0; i < outputs.size(); ++i) {
    JUST(JUST(outputs.at(i)->current_grad())->PushPartialTensor(out_grads.at(i)));
  }
  // Runs each FunctionNode
  for (const auto& weak_func_node : node_list_) {
    const auto& func_node = weak_func_node.lock();
    CHECK_NOTNULL_OR_RETURN(func_node);
    if (JUST(func_node->Apply(create_graph))) {
      JUST(func_node->AccGrad4RetainGradTensor());
      func_node->ReleaseOutTensorArgs();
      if (!retain_graph) { func_node->ReleaseData(); }
    }
  }
  // Gets input grads and resume retain_grad
  for (int i = 0; i < inputs.size(); ++i) {
    input_current_grad->at(i) = JUST(inputs.at(i)->acc_grad());
    if (!ori_retain_grad.at(i)) {
      JUST(inputs.at(i)->set_acc_grad(nullptr));
      JUST(inputs.at(i)->set_retain_grad(false));
    }
  }
  if (!retain_graph) { ClearEngine(); }
  return input_current_grad;
}

Maybe<FunctionNode> StackAutogradEngine::AddBackwardFuncPtr(
    const std::string& op_type_name,
    const std::shared_ptr<const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>&
        backward_fn,
    const TensorTuple& inputs, TensorTuple* outputs) {
  // Firstly push function_node of tensor in stack which is leaf and requires_grad
  for (const std::shared_ptr<Tensor>& in_tensor : inputs) {
    if (in_tensor->is_leaf() && in_tensor->requires_grad()) {
      if (!in_tensor->grad_fn_node()) { JUST(AddAccumulateFunctionNode(in_tensor)); }
      StackFunctionNode* stack_function_node =
          dynamic_cast<StackFunctionNode*>(in_tensor->mut_grad_fn_node().get());
      if (!stack_function_node->is_in_stack()) {
        stack_function_node->set_is_in_stack(true);
        node_list_.push_front(in_tensor->mut_grad_fn_node());
      }
    }
  }

  std::shared_ptr<StackFunctionNode> func_node =
      std::make_shared<StackFunctionNode>(op_type_name, backward_fn, inputs, *outputs);
  for (const std::shared_ptr<Tensor>& out_tensor : *outputs) {
    out_tensor->set_grad_fn_node(func_node);
  }
  func_node->set_is_in_stack(true);
  node_list_.push_front(func_node);
  return std::static_pointer_cast<FunctionNode>(func_node);
}

void GraphFunctionNode::ReleaseData() {
  if (!input_meta_data_.empty()) { backward_fn_.reset(); }
}

GraphFunctionNode::GraphFunctionNode(
    const std::string& op_type_name,
    const std::shared_ptr<const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>&
        backward_fn,
    const TensorTuple& inputs, const TensorTuple& outputs)
    : FunctionNode(op_type_name) {
  input_meta_data_.resize(inputs.size());
  next_functions_->reserve(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    if (inputs.at(i)->requires_grad()) {
      input_meta_data_.at(i) = inputs.at(i)->mut_autograd_meta();
      next_functions_->emplace_back(inputs.at(i)->mut_grad_fn_node());
    }
  }

  output_meta_data_.resize(outputs.size());
  output_tensor_infos_.reserve(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    const auto& autograd_meta =
        NewAutogradMeta(outputs.at(i)->requires_grad(), outputs.at(i)->is_leaf());
    outputs.at(i)->set_autograd_meta(autograd_meta);
    output_meta_data_.at(i) = outputs.at(i)->mut_autograd_meta();
    output_tensor_infos_.emplace_back(TensorInfo(*outputs.at(i)));
  }

  backward_fn_ = backward_fn;
}

GraphTask::GraphTask(const TensorTuple& outputs, bool retain_graph, bool create_graph)
    : retain_graph_(retain_graph), create_graph_(create_graph) {
  roots_.reserve(outputs.size());
  for (const auto& out_tensor : outputs) {
    FunctionNode* node = out_tensor->mut_grad_fn_node().get();
    roots_.emplace_back(node);
    dependencies_.insert(std::make_pair(node, 0));
  }
}

// Computes the number of dependencies for each FunctionNode
Maybe<void> GraphTask::ComputeDependencies() {
  HashSet<FunctionNode*> seen;
  std::stack<FunctionNode*> stack;
  for (FunctionNode* node : roots_) { stack.push(node); }

  while (!stack.empty()) {
    FunctionNode* node = stack.top();
    stack.pop();
    if (/*bool has_seen=*/!seen.insert(node).second) { continue; }
    for (const auto& next_grad_fn : *(node->GetNextFunctions())) {
      FunctionNode* next_node = next_grad_fn.get();
      dependencies_[next_node] += 1;
      if (seen.find(next_node) == seen.end()) { stack.push(next_node); }
    }
  }
  return Maybe<void>::Ok();
}

// Computes the number of dependencies for each FunctionNode and prunes useless FunctionNode
// according to input tensors
Maybe<void> GraphTask::ComputeDependenciesAndPruneNode(const TensorTuple& inputs) {
  struct NodeFrame {
    NodeFrame(FunctionNode* node) : node_(node), next_function_idx_(0) {}
    FunctionNode* node_;
    size_t next_function_idx_;

    FunctionNode* GetNextFunction() {
      if (next_function_idx_ < node_->GetNextFunctions()->size()) {
        next_function_idx_ += 1;
        return node_->GetNextFunctions()->at(next_function_idx_ - 1).get();
      } else {
        return nullptr;
      }
    }
  };

  for (const auto& input : inputs) {
    CHECK_NOTNULL_OR_RETURN(input->mut_grad_fn_node().get());
    need_execute_.insert(input->mut_grad_fn_node().get());
  }

  HashSet<FunctionNode*> seen;
  std::stack<NodeFrame> stack;

  // Note: dfs to determine each FunctionNode should execute or not.
  for (const auto& root : roots_) { stack.push(NodeFrame(root)); }
  while (!stack.empty()) {
    NodeFrame& frame = stack.top();
    if (/*bool has_seen=*/seen.find(frame.node_) != seen.end()) {
      stack.pop();
      continue;
    }
    if (FunctionNode* node = frame.GetNextFunction()) {
      dependencies_[node] += 1;
      if (seen.find(node) == seen.end()) {
        stack.push(NodeFrame(node));
        continue;  // recurse
      }
    } else {
      bool need_execute = std::any_of(frame.node_->GetNextFunctions()->begin(),
                                      frame.node_->GetNextFunctions()->end(),
                                      [&](const std::shared_ptr<FunctionNode>& fn) {
                                        return need_execute_.find(fn.get()) != need_execute_.end();
                                      });
      if (need_execute) { need_execute_.insert(frame.node_); }
      seen.insert(frame.node_);
      stack.pop();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> GraphTask::Apply(bool save_grad_for_leaf) {
  std::queue<FunctionNode*> queue;
  for (FunctionNode* node : roots_) {
    if (dependencies_[node] == 0) { queue.push(node); }
  }

  while (!queue.empty()) {
    FunctionNode* node = queue.front();
    queue.pop();
    if (!need_execute_.empty() && need_execute_.find(node) == need_execute_.end()) {
      node->ReleaseOutTensorArgs();
      continue;
    }
    if (/*bool not_ready_to_apply=*/!(JUST(node->Apply(create_graph_)))) { continue; }
    if (save_grad_for_leaf) { JUST(node->AccGrad4LeafTensor(create_graph_)); }
    JUST(node->AccGrad4RetainGradTensor());
    node->ReleaseOutTensorArgs();
    if (!retain_graph_) { node->ReleaseData(); }

    for (const auto& next_grad_fn : *(node->GetNextFunctions())) {
      FunctionNode* next_node = next_grad_fn.get();
      dependencies_[next_node] -= 1;
      if (dependencies_[next_node] == 0) { queue.push(next_node); }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> GraphAutogradEngine::RunBackwardAndSaveGrads4LeafTensor(const TensorTuple& outputs,
                                                                    const TensorTuple& out_grads,
                                                                    bool retain_graph,
                                                                    bool create_graph) {
  for (int i = 0; i < outputs.size(); ++i) {
    JUST(JUST(outputs.at(i)->current_grad())->PushPartialTensor(out_grads.at(i)));
  }
  GraphTask graph_task(outputs, retain_graph, create_graph);
  JUST(graph_task.ComputeDependencies());
  JUST(graph_task.Apply(/*save_grad_for_leaf=*/true));
  return Maybe<void>::Ok();
}

Maybe<TensorTuple> GraphAutogradEngine::RunBackwardAndReturnInputsTensorGrad(
    const TensorTuple& outputs, const TensorTuple& inputs, const TensorTuple& out_grads,
    bool retain_graph, bool create_graph) {
  std::shared_ptr<TensorTuple> input_current_grad = std::make_shared<TensorTuple>(inputs.size());
  GraphTask graph_task(outputs, retain_graph, create_graph);
  std::vector<bool> ori_retain_grad(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    ori_retain_grad.at(i) = inputs.at(i)->retain_grad();
    JUST(inputs.at(i)->set_retain_grad(true));
  }
  for (int i = 0; i < outputs.size(); ++i) {
    JUST(JUST(outputs.at(i)->current_grad())->PushPartialTensor(out_grads.at(i)));
  }

  JUST(graph_task.ComputeDependenciesAndPruneNode(inputs));
  JUST(graph_task.Apply(/*save_grad_for_leaf=*/false));

  // Gets input grads and resume retain_grad
  for (int i = 0; i < inputs.size(); ++i) {
    input_current_grad->at(i) = JUST(inputs.at(i)->acc_grad());
    if (!ori_retain_grad.at(i)) {
      JUST(inputs.at(i)->set_acc_grad(nullptr));
      JUST(inputs.at(i)->set_retain_grad(false));
    }
  }
  return input_current_grad;
}

Maybe<FunctionNode> GraphAutogradEngine::AddBackwardFuncPtr(
    const std::string& op_type_name,
    const std::shared_ptr<const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>&
        backward_fn,
    const TensorTuple& inputs, TensorTuple* outputs) {
  // Firstly push function_node of tensor in stack which is leaf and requires_grad
  for (const std::shared_ptr<Tensor>& in_tensor : inputs) {
    if (in_tensor->is_leaf() && in_tensor->requires_grad()) {
      if (!in_tensor->grad_fn_node()) { JUST(AddAccumulateFunctionNode(in_tensor)); }
    }
  }

  std::shared_ptr<FunctionNode> func_node =
      std::make_shared<GraphFunctionNode>(op_type_name, backward_fn, inputs, *outputs);
  for (const std::shared_ptr<Tensor>& out_tensor : *outputs) {
    out_tensor->set_grad_fn_node(func_node);
  }
  return func_node;
}

AutogradEngine* GetThreadLocalAutogradEngine() {
  // thread_local static StackAutogradEngine autograd_engine;
  thread_local static GraphAutogradEngine autograd_engine;
  return &autograd_engine;
}

Maybe<void> AddAccumulateFunctionNode(const std::shared_ptr<Tensor>& tensor) {
  auto backward_fn =
      std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
          [=](const TensorTuple& out_grads, TensorTuple* in_grads,
              bool create_graph) -> Maybe<void> { return Maybe<void>::Ok(); });
  tensor->set_grad_fn_node(std::make_shared<GraphFunctionNode>(
      "accumulate_grad", backward_fn, TensorTuple(), TensorTuple({tensor})));
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
