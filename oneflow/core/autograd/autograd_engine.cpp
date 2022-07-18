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

#include <memory>
#include <stack>
#include <queue>
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_arg.h"
#include "oneflow/core/framework/tensor_methods.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/global_param_grad_sync_mode.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/common/env_var/autograd.h"

namespace oneflow {
namespace one {

namespace {

void GatherFunctionNodes(FunctionNode* node, std::stack<std::shared_ptr<FunctionNode>>& stack) {
  for (auto& prev_node : node->next_functions()) {
    if (prev_node) {
      if (prev_node.use_count() == 1) { stack.push(prev_node); }
    }
  }
}

/* NOTE:
 * Stack overflows when releasing a very deep computation graph without
 * a custom deleter.
 *
 * For example, here is a very deep computation graph:
 * Tensor -> FunctionNode -> Tensor -> FunctionNode -> ... -> Tensor -> FunctionNode
 * When releasing the first Tensor, it will trigger the recursive deletion and stack overflow.
 *
 * So we must set a custom deleter and release them iteratively.
 */
void FunctionNodeDeleter(FunctionNode* node) {
  std::stack<std::shared_ptr<FunctionNode>> stack;
  node->ReleaseData();
  GatherFunctionNodes(node, stack);
  delete node;

  while (!stack.empty()) {
    auto now_node = std::move(stack.top());
    stack.pop();
    now_node->ReleaseData();
    GatherFunctionNodes(now_node.get(), stack);
  }
}

bool IsReadyToRun(const std::vector<std::shared_ptr<AutogradMeta>>& out_meta_datas) {
  return std::any_of(out_meta_datas.begin(), out_meta_datas.end(),
                     [](const std::shared_ptr<AutogradMeta>& meta_data) {
                       return !meta_data->current_grad()->Empty();
                     });
}

Maybe<void> CopyOrAccGrad(AutogradMeta* autograd_meta, bool autograd_mode) {
  autograd::AutoGradMode mode(autograd_mode);
  auto current_grad = JUST(autograd_meta->current_grad()->GetAccTensor({}));
  if (!current_grad) { return Maybe<void>::Ok(); }
  if (autograd_meta->acc_grad()) {
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

Maybe<void> RawTorchGlobalTensor(const std::shared_ptr<one::Tensor>& tensor) {
  // Do nothing.
  return Maybe<void>::Ok();
}

static constexpr auto* TorchGlobalTensor = DECORATE(&RawTorchGlobalTensor, CheckGlobalTensorMeta);

Maybe<void> CheckGlobalTensorsMeta(const TensorTuple& tensor_tuple) {
  for (const auto& tensor : tensor_tuple) {
    if (tensor->is_global()) { JUST(TorchGlobalTensor(tensor)); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> TouchInTmpComputeStream(const TensorTuple& inputs) {
  for (auto input : inputs) {
    if (input->is_global()) { input = JUST(input->cur_rank_phy_tensor()); }
    if (input) {
      Symbol<Device> device = JUST(input->device());
      auto stream = JUST(Stream::New(device, StreamRole::kTmpCompute));
      JUST(Touch(input, stream));
    }
  }
  return Maybe<void>::Ok();
}

constexpr static int kSmallTensorThreshold = 1024;

Maybe<TensorTuple> TryCopyForSmallTensor(const TensorTuple& inputs) {
  auto outputs = std::make_shared<TensorTuple>();
  outputs->reserve(inputs.size());
  for (auto input : inputs) {
    if (input->shape()->elem_cnt() <= kSmallTensorThreshold) {
      input = JUST(functional::Identity(input));
    }
    outputs->push_back(input);
  }
  return outputs;
}

}  // namespace

Maybe<void> AutogradEngine::RunBackwardAndSaveGrads4LeafTensorIf(const TensorTuple& outputs,
                                                                 const TensorTuple& out_grads,
                                                                 bool retain_graph,
                                                                 bool create_graph) {
  JUST(CheckGlobalTensorsMeta(outputs));
  JUST(CheckGlobalTensorsMeta(out_grads));
  DisableCheckGlobalTensorMetaScope disable_meta_check;
  if (ThreadLocalEnvBool<ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM>()) {
    // Put outputs into kTmpCompute stream for reducing blocking time of outputs[i].numpy() in main
    // thread.
    auto copied_outputs = JUST(TryCopyForSmallTensor(outputs));
    JUST(TouchInTmpComputeStream(outputs));
    return RunBackwardAndSaveGrads4LeafTensor(*copied_outputs, out_grads, retain_graph,
                                              create_graph);
  } else {
    return RunBackwardAndSaveGrads4LeafTensor(outputs, out_grads, retain_graph, create_graph);
  }
}

Maybe<TensorTuple> AutogradEngine::RunBackwardAndReturnInputsTensorGradIf(
    const TensorTuple& outputs, const TensorTuple& inputs, const TensorTuple& out_grads,
    bool retain_graph, bool create_graph) {
  JUST(CheckGlobalTensorsMeta(outputs));
  JUST(CheckGlobalTensorsMeta(inputs));
  JUST(CheckGlobalTensorsMeta(out_grads));
  DisableCheckGlobalTensorMetaScope disable_meta_check;
  return RunBackwardAndReturnInputsTensorGrad(outputs, inputs, out_grads, retain_graph,
                                              create_graph);
}

Maybe<void> FunctionNode::AccGrad4RetainGradTensor() {
  for (const std::shared_ptr<AutogradMeta>& out : output_meta_data_) {
    if (out->retain_grad()) { JUST(CopyOrAccGrad(out.get(), /*autograd_mode=*/false)); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> FunctionNode::AccGrad4LeafTensor(bool create_graph) {
  for (auto i = 0; i < output_meta_data_.size(); i++) {
    auto& out = output_meta_data_[i];

    if (out->is_leaf() && out->requires_grad()) {
      JUST(CopyOrAccGrad(out.get(), /*autograd_mode=*/false));

      // control acc_grad to do boxing conditionally
      const auto& acc_grad = out->acc_grad();
      if (GlobalGradSyncMode::is_enabled() && acc_grad->is_global()) {
        auto& tensor_info = output_tensor_infos_[i];
        const auto& placement = JUST(tensor_info.placement());
        const auto& nd_sbp = JUST(tensor_info.sbp());
        JUST(out->set_acc_grad(
            JUST(functional::ToGlobal(acc_grad, placement, *JUST(GetSbpList(nd_sbp)),
                                      GetNoneSbpList(), /* check_meta */ false))));
      }
    }
  }
  return Maybe<void>::Ok();
}

void FunctionNode::ReleaseOutTensorArgs() {
  for (const std::shared_ptr<AutogradMeta>& meta_data : output_meta_data_) {
    meta_data->current_grad()->Release();
  }
}

Maybe<bool> FunctionNode::Apply(bool create_graph) {
  CHECK_NOTNULL_OR_RETURN(backward_fn_)
      << "This FunctionNode with name `" << name() << "` has been released.\n"
      << "Maybe you try to backward through the node a second time. Specify retain_graph=True when "
         "calling .backward() or autograd.grad() the first time.";
  if (!IsReadyToRun(output_meta_data_)) { return false; }
  TensorTuple input_grads(input_meta_data_.size());
  TensorTuple output_grads(output_meta_data_.size());
  for (int i = 0; i < output_meta_data_.size(); ++i) {
    if (output_meta_data_.at(i)->current_grad()->Empty()) {
      output_grads.at(i) = JUST(output_tensor_infos_.at(i).zeros());
    } else {
      const auto& hooks = JUST(oneflow::VectorAt(output_meta_data_, i))->hooks();
      JUST(oneflow::VectorAt(output_grads, i)) =
          JUST(JUST(oneflow::VectorAt(output_meta_data_, i))->current_grad()->GetAccTensor(hooks));
    }
  }
  JUST(backward_fn_->body(output_grads, &input_grads, create_graph));
  for (int i = 0; i < input_meta_data_.size(); ++i) {
    if (JUST(VectorAt(input_grads, i))) {
      CHECK_NOTNULL_OR_RETURN(input_meta_data_[i])
          << name_
          << " calculate grad for tensor which requires_grad is False. Please submit an issue in "
             "`https://github.com/Oneflow-Inc/oneflow/issues` and we will fix it as soon as "
             "possible";
      JUST(input_meta_data_[i]->current_grad()->PushPartialTensor(JUST(VectorAt(input_grads, i))));
    } else {
      CHECK_OR_RETURN(!input_meta_data_[i])
          << name() << "'s input[" << i
          << "] need calculate grad but got nullptr. Please submit an issue in "
             "`https://github.com/Oneflow-Inc/oneflow/issues` and we will fix it as soon as "
             "possible;";
    }
  }
  return true;
}

void GraphFunctionNode::ReleaseData() {
  if (backward_fn_ && backward_fn_->status()) { backward_fn_.reset(); }
}

/*static*/ std::shared_ptr<GraphFunctionNode> GraphFunctionNode::New(
    const std::string& name, const std::shared_ptr<BackwardFunction>& backward_fn,
    const TensorTuple& inputs, const TensorTuple& outputs) {
  auto node = std::shared_ptr<GraphFunctionNode>(
      new GraphFunctionNode(name, backward_fn, inputs, outputs), FunctionNodeDeleter);
  return node;
}

GraphFunctionNode::GraphFunctionNode(const std::string& name,
                                     const std::shared_ptr<BackwardFunction>& backward_fn,
                                     const TensorTuple& inputs, const TensorTuple& outputs)
    : FunctionNode(name, backward_fn) {
  input_meta_data_.resize(inputs.size());
  next_functions_.reserve(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    if (inputs.at(i)->requires_grad()) {
      input_meta_data_.at(i) = inputs.at(i)->mut_autograd_meta();
      next_functions_.emplace_back(inputs.at(i)->mut_grad_fn_node());
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
    for (const auto& next_grad_fn : node->next_functions()) {
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
    explicit NodeFrame(FunctionNode* node) : node_(node), next_function_idx_(0) {}
    FunctionNode* node_;
    size_t next_function_idx_;

    FunctionNode* GetNextFunction() {
      if (next_function_idx_ < node_->next_functions().size()) {
        next_function_idx_ += 1;
        return node_->next_functions().at(next_function_idx_ - 1).get();
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
      bool need_execute =
          std::any_of(frame.node_->next_functions().begin(), frame.node_->next_functions().end(),
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

    for (const auto& next_grad_fn : node->next_functions()) {
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

Maybe<FunctionNode> GraphAutogradEngine::AddNode(
    const std::string& name, const std::shared_ptr<BackwardFunction>& backward_fn,
    const TensorTuple& inputs, TensorTuple* outputs) {
  OF_PROFILER_RANGE_PUSH("AddAccumulateFunctionNode");
  // Firstly push function_node of tensor in stack which is leaf and requires_grad
  for (const std::shared_ptr<Tensor>& in_tensor : inputs) {
    if (in_tensor->is_leaf() && in_tensor->requires_grad()) {
      if (!in_tensor->grad_fn_node()) { JUST(AddAccumulateFunctionNode(in_tensor)); }
    }
  }

  OF_PROFILER_RANGE_POP();
  OF_PROFILER_RANGE_PUSH("set_grad_fn_node");
  std::shared_ptr<FunctionNode> func_node =
      GraphFunctionNode::New(name, backward_fn, inputs, *outputs);
  for (const std::shared_ptr<Tensor>& out_tensor : *outputs) {
    out_tensor->set_grad_fn_node(func_node);
  }
  OF_PROFILER_RANGE_POP();
  return func_node;
}

AutogradEngine* GetThreadLocalAutogradEngine() {
  thread_local static GraphAutogradEngine autograd_engine;
  return &autograd_engine;
}

Maybe<void> AddAccumulateFunctionNode(const std::shared_ptr<Tensor>& tensor) {
  auto backward_fn = std::make_shared<BackwardFunction>();
  backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                          bool create_graph) -> Maybe<void> { return Maybe<void>::Ok(); };
  backward_fn->status = []() { return false; };
  tensor->set_grad_fn_node(GraphFunctionNode::New(
      "accumulate_grad", backward_fn, /*inputs=*/TensorTuple{}, /*outputs*/ TensorTuple{tensor}));
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
