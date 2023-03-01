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
#include "fmt/core.h"
#include "fmt/format.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_arg.h"
#include "oneflow/core/framework/tensor_methods.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/global_param_grad_sync_mode.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

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
  auto current_grad = JUST(autograd_meta->current_grad_value());
  if (!current_grad) { return Maybe<void>::Ok(); }
  if (autograd_meta->acc_grad()) {
    JUST(functional::Add(autograd_meta->acc_grad(), current_grad, /*alpha=*/1.0,
                         /*inplace=*/true));
  } else {
    // NOTE: acc_grad can not share data with current_grad, because accumulate acc_grad
    // with inplace operation and it maybe change current_grad to get wrong result.
    // See more details in https://github.com/Oneflow-Inc/oneflow/issues/8248
    if (!LazyMode::is_enabled()) { current_grad = JUST(functional::Identity(current_grad)); }
    JUST(autograd_meta->set_acc_grad(current_grad));
  }
  for (const auto& hook : autograd_meta->post_grad_accumulation_hooks()) {
    auto new_grad = hook(autograd_meta->acc_grad());
    if (new_grad) { JUST(autograd_meta->set_acc_grad(new_grad)); }
  }

  return Maybe<void>::Ok();
}

Maybe<void> RawTouchGlobalTensor(const std::shared_ptr<one::Tensor>& tensor) {
  // Do nothing.
  return Maybe<void>::Ok();
}

static constexpr auto* TouchGlobalTensor = DECORATE(&RawTouchGlobalTensor, CheckGlobalTensorMeta);

Maybe<void> CheckGlobalTensorsMeta(const TensorTuple& tensor_tuple) {
  for (const auto& tensor : tensor_tuple) {
    if (tensor->is_global() && tensor->is_eager()) { JUST(TouchGlobalTensor(tensor)); }
  }
  return Maybe<void>::Ok();
}

std::string GetDebugGraphFileName(const std::string& mode, const std::string& suffix) {
  return fmt::format("autograd_{}_rank{}_suffix_graph.dot", mode, GlobalProcessCtx::Rank(), suffix);
}

}  // namespace

Maybe<void> AutogradEngine::RunBackwardAndSaveGrads4LeafTensorIf(const TensorTuple& outputs,
                                                                 const TensorTuple& out_grads,
                                                                 bool retain_graph,
                                                                 bool create_graph) {
  JUST(CheckGlobalTensorsMeta(outputs));
  JUST(CheckGlobalTensorsMeta(out_grads));
  DisableCheckGlobalTensorMetaScope disable_meta_check;
  return RunBackwardAndSaveGrads4LeafTensor(outputs, out_grads, retain_graph, create_graph);
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

Maybe<void> FunctionNode::AccGrad4RetainGradTensor(bool create_graph) {
  for (const std::shared_ptr<AutogradMeta>& out : output_meta_data_) {
    if (out->retain_grad()) { JUST(CopyOrAccGrad(out.get(), create_graph)); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> FunctionNode::AccGrad4LeafTensor(bool create_graph) {
  for (auto i = 0; i < output_meta_data_.size(); i++) {
    auto& out = output_meta_data_[i];

    if (out->is_leaf() && out->requires_grad()) {
      JUST(CopyOrAccGrad(out.get(), /*autograd_mode=*/create_graph));

      // control acc_grad to do boxing conditionally
      const auto& acc_grad = out->acc_grad();
      if (!LazyMode::is_enabled() && GlobalGradSyncMode::is_enabled() && acc_grad->is_global()
          && acc_grad->is_eager()) {
        auto& tensor_info = output_tensor_infos_[i];
        const auto& placement = JUST(tensor_info.placement());
        const auto& nd_sbp = JUST(tensor_info.sbp());
        JUST(out->set_acc_grad(
            JUST(functional::ToGlobal(acc_grad, placement, *JUST(GetSbpList(nd_sbp)),
                                      GetNoneSbpList(), /* check_meta */ false, /*copy=*/false))));
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
    if (output_meta_data_[i]->current_grad()->Empty()) {
      // Only initialize out_grads for those requires_grad outputs
      if (output_meta_data_[i]->requires_grad()) {
        output_grads[i] = JUST(output_tensor_infos_[i].zeros());
      }
    } else {
      JUST(oneflow::VectorAt(output_grads, i)) =
          JUST(JUST(oneflow::VectorAt(output_meta_data_, i))->current_grad_value());
    }
  }
  JUST(backward_fn_->body(output_grads, &input_grads, create_graph));
  for (const auto& hook : hooks_) {
    auto new_input_grads = hook(input_grads, output_grads);
    if (new_input_grads.has_value()) {
      auto new_input_grads_value = *JUST(new_input_grads);
      CHECK_EQ_OR_RETURN(new_input_grads_value.size(), input_grads.size())
          << "The number of input grads returned by hook is not correct, expected "
          << input_grads.size() << ", but got " << new_input_grads_value.size() << ".";
      for (int i = 0; i < input_grads.size(); ++i) { input_grads[i] = new_input_grads_value[i]; }
    }
  }
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
    output_tensor_infos_.emplace_back(*outputs.at(i));
  }

  backward_fn_ = backward_fn;
}

GraphTask::GraphTask(const TensorTuple& outputs, bool retain_graph, bool create_graph)
    : retain_graph_(retain_graph), create_graph_(create_graph) {
  roots_.reserve(outputs.size());
  for (const auto& out_tensor : outputs) {
    FunctionNode* node = out_tensor->mut_grad_fn_node().get();
    roots_.emplace_back(node);
  }
}

Maybe<void> GraphTask::WriteGraphToDotFile(const std::string& file_name) const {
  auto ExecInfoToDotString = [](const ExecInfo& exec_info) -> std::string {
    std::stringstream ss;
    ss << "ExecInfo{\\l";
    ss << "\tdependencies: " << exec_info.dependencies << "\\l";
    ss << "\tneed_execute: " << exec_info.need_execute << "\\l";
    if (exec_info.capture_indices) {
      ss << "\tcapture_indices: [";
      for (const auto& out_idx_and_capture_idx : *exec_info.capture_indices) {
        ss << out_idx_and_capture_idx.second << ", ";
      }
      ss << "]\\l";
    }
    ss << "}\\l";
    return ss.str();
  };

  auto log_stream = TeePersistentLogStream::Create(file_name);
  std::vector<std::string> lines;
  lines.emplace_back("digraph AutogradTaskGraph {");
  lines.emplace_back("\tmargin=\"1.5\";");
  lines.emplace_back("\tnode [shape=box];");
  for (auto iter = grad_fn2exec_info_.begin(); iter != grad_fn2exec_info_.end(); ++iter) {
    const FunctionNode* node = iter->first;
    const ExecInfo& exec_info = iter->second;
    // write label attribute
    std::string node_color = "black";
    if (exec_info.dependencies == 0 && exec_info.need_execute) {  // start node
      node_color = "red";
    } else if (exec_info.need_execute && exec_info.capture_indices) {  // end node
      node_color = "green";
    }
    lines.emplace_back(fmt::format(
        "\t\"{}\" [label=\"{}\\l{}\\l{}\", color={}];", static_cast<const void*>(node),
        node->name(), static_cast<const void*>(node), ExecInfoToDotString(exec_info), node_color));
    // write edge
    for (const auto& next_fn : node->next_functions()) {
      lines.emplace_back(fmt::format("\t\"{}\" -> \"{}\";", static_cast<const void*>(node),
                                     static_cast<const void*>(next_fn.get())));
    }
  }
  lines.emplace_back("}");
  log_stream << fmt::format("{}", fmt::join(lines, "\n"));
  log_stream->Flush();
  return Maybe<void>::Ok();
}

// Computes the number of dependencies for each FunctionNode
Maybe<void> GraphTask::ComputeDependencies() {
  HashSet<FunctionNode*> seen;
  std::stack<FunctionNode*> stack;
  for (FunctionNode* node : roots_) {
    stack.push(node);
    grad_fn2exec_info_[node].need_execute = true;
  }

  while (!stack.empty()) {
    FunctionNode* node = stack.top();
    stack.pop();
    if (/*bool has_seen=*/!seen.insert(node).second) { continue; }
    for (const auto& next_grad_fn : node->next_functions()) {
      FunctionNode* next_node = next_grad_fn.get();
      ExecInfo& exec_info = grad_fn2exec_info_[next_node];
      exec_info.dependencies += 1;
      exec_info.need_execute = true;
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

  // initialize all variable to capture grad for input tensors
  captured_grads_ = std::make_shared<TensorTuple>(inputs.size());
  for (int idx = 0; idx < inputs.size(); idx++) {
    const auto& input = inputs[idx];
    CHECK_NOTNULL_OR_RETURN(input->mut_grad_fn_node().get());  //  NOLINT(maybe-need-error-msg)
    ExecInfo& exec_info = grad_fn2exec_info_[input->mut_grad_fn_node().get()];
    exec_info.need_execute = true;
    if (!exec_info.capture_indices) {
      exec_info.capture_indices = std::make_unique<std::vector<std::pair<size_t, size_t>>>();
    }
    exec_info.capture_indices->emplace_back(std::make_pair(input->get_grad_fn_output_index(), idx));
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
      grad_fn2exec_info_[node].dependencies += 1;
      if (seen.find(node) == seen.end()) {
        stack.push(NodeFrame(node));
        continue;  // recurse
      }
    } else {
      grad_fn2exec_info_[frame.node_].need_execute |=
          std::any_of(frame.node_->next_functions().begin(), frame.node_->next_functions().end(),
                      [&](const std::shared_ptr<FunctionNode>& fn) {
                        return grad_fn2exec_info_[fn.get()].need_execute;
                      });
      seen.insert(frame.node_);
      stack.pop();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> GraphTask::Apply(bool save_grad_for_leaf) {
  std::queue<FunctionNode*> queue;
  for (FunctionNode* node : roots_) {
    if (grad_fn2exec_info_[node].dependencies == 0) { queue.push(node); }
  }

  while (!queue.empty()) {
    FunctionNode* node = queue.front();
    queue.pop();
    auto& exec_info = grad_fn2exec_info_[node];

    if (!exec_info.need_execute) {
      node->ReleaseOutTensorArgs();
      continue;
    }
    BackwardPassScopeGuard backward_guard(node->scope());
    if (/*bool not_ready_to_apply=*/!(JUST(node->Apply(create_graph_)))) { continue; }
    if (exec_info.capture_indices) {
      CHECK_NOTNULL_OR_RETURN(captured_grads_.get()) << "captured grads in GraphTask is nullptr";
      for (const auto& out_idx_and_capture_idx : *exec_info.capture_indices) {
        JUST(VectorAt(*captured_grads_, out_idx_and_capture_idx.second)) =
            JUST(JUST(VectorAt(node->output_meta_data_, out_idx_and_capture_idx.first))
                     ->current_grad_value());
      }
    }
    if (save_grad_for_leaf) { JUST(node->AccGrad4LeafTensor(create_graph_)); }
    JUST(node->AccGrad4RetainGradTensor(create_graph_));
    node->ReleaseOutTensorArgs();
    if (!retain_graph_) { node->ReleaseData(); }

    for (const auto& next_grad_fn : node->next_functions()) {
      FunctionNode* next_node = next_grad_fn.get();
      int32_t& dependencies = grad_fn2exec_info_[next_node].dependencies;
      dependencies -= 1;
      if (dependencies == 0) { queue.push(next_node); }
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
  if (IsInDebugMode()) {
    JUST(
        graph_task.WriteGraphToDotFile(GetDebugGraphFileName("backward", std::to_string(clock()))));
  }
  JUST(graph_task.Apply(/*save_grad_for_leaf=*/true));
  return Maybe<void>::Ok();
}

Maybe<TensorTuple> GraphAutogradEngine::RunBackwardAndReturnInputsTensorGrad(
    const TensorTuple& outputs, const TensorTuple& inputs, const TensorTuple& out_grads,
    bool retain_graph, bool create_graph) {
  for (int i = 0; i < outputs.size(); ++i) {
    JUST(JUST(outputs.at(i)->current_grad())->PushPartialTensor(out_grads.at(i)));
  }

  GraphTask graph_task(outputs, retain_graph, create_graph);
  JUST(graph_task.ComputeDependenciesAndPruneNode(inputs));
  if (IsInDebugMode()) {
    JUST(graph_task.WriteGraphToDotFile(GetDebugGraphFileName("grad", std::to_string(clock()))));
  }
  JUST(graph_task.Apply(/*save_grad_for_leaf=*/false));
  return graph_task.GetCapturedGrads();
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
  for (int i = 0; i < outputs->size(); ++i) {
    const std::shared_ptr<Tensor>& out_tensor = JUST(VectorAt(*outputs, i));
    out_tensor->set_grad_fn_node(func_node);
    out_tensor->set_grad_fn_output_index(i);
  }
  if (LazyMode::is_enabled()) { func_node->set_scope(JUST(GetCurrentScope())); }
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
  tensor->set_grad_fn_output_index(0);
  if (LazyMode::is_enabled()) {
    tensor->mut_grad_fn_node()->set_scope(JUST(GetTensorScope(tensor)));
  }
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
