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

#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_arg.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/autograd/autograd_mode.h"

namespace oneflow {
namespace one {

namespace {

bool IsReadyToRun(const std::vector<std::shared_ptr<TensorArg>>& out_grads) {
  return std::any_of(
      out_grads.begin(), out_grads.end(),
      [](const std::shared_ptr<TensorArg>& tensor_arg) { return !tensor_arg->Empty(); });
}

Maybe<void> InitEmptyTensorArgs2ZerosTensor(const TensorTuple& outputs,
                                            std::vector<std::shared_ptr<TensorArg>>& out_grads) {
  const auto& zero_like = JUST(op_expr_helper::ZeroLikeOp());
  for (int i = 0; i < out_grads.size(); ++i) {
    if (out_grads.at(i)->Empty()) {
      TensorTuple output(1);
      JUST(JUST(OpInterpUtil::GetInterpreter())->Apply(*zero_like, {outputs.at(i)}, &output));
      JUST(out_grads.at(i)->PushPartialTensor(output.at(0)));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> CopyOrAccGrad(Tensor* tensor, bool autograd_mode) {
  autograd::AutoGradMode mode(autograd_mode);
  const auto& now_grad = JUST(tensor->now_grad_arg()->GetAccTensor());
  if (tensor->acc_grad()) {
    TensorTuple input = {tensor->acc_grad(), now_grad};
    TensorTuple output(1);
    const auto& add = JUST(op_expr_helper::AddOp());
    JUST(JUST(OpInterpUtil::GetInterpreter())->Apply(*add, input, &output));
    tensor->set_acc_grad(output.at(0));
  } else {
    tensor->set_acc_grad(now_grad);
  }
  return Maybe<void>::Ok();
}

}  // namespace

StackFunctionNode::StackFunctionNode(
    const std::shared_ptr<const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>&
        backward_fn,
    const TensorTuple& inputs, const TensorTuple& outputs) {
  inputs_ = std::make_shared<TensorTuple>(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    inputs_->at(i) = inputs.at(i);
    in_grads_.emplace_back(inputs.at(i)->now_grad_arg());
  }

  outputs_ = std::make_shared<TensorTuple>(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    outputs_->at(i) = outputs.at(i)->detach();
    out_grads_.emplace_back(outputs.at(i)->now_grad_arg());
  }

  backward_fn_ = backward_fn;
  is_in_stack_ = false;
}

Maybe<void> StackFunctionNode::AccGrad4RetainGradTensor() {
  for (int i = 0; i < outputs_->size(); ++i) {
    if (outputs_->at(i)->retain_grad() && outputs_->at(i)->requires_grad()) {
      JUST(CopyOrAccGrad(outputs_->at(i).get(), /*autograd_mode=*/false));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> StackFunctionNode::AccGrad4LeafTensor(bool create_graph) {
  for (int i = 0; i < outputs_->size(); ++i) {
    if (outputs_->at(i)->is_leaf() && outputs_->at(i)->requires_grad()) {
      JUST(CopyOrAccGrad(outputs_->at(i).get(), /*autograd_mode=*/create_graph));
    }
  }
  return Maybe<void>::Ok();
}

void StackFunctionNode::ReleaseOutTensorArgs() {
  for (const std::shared_ptr<TensorArg>& tensor_arg : out_grads_) { tensor_arg->Release(); }
}

void StackFunctionNode::ReleaseData() {
  if (!inputs_->empty()) {
    inputs_.reset();
    outputs_.reset();
    in_grads_.clear();
    out_grads_.clear();
    backward_fn_.reset();
  }
  is_in_stack_ = false;
}

Maybe<bool> StackFunctionNode::Apply(bool create_graph) {
  CHECK_NOTNULL_OR_RETURN(backward_fn_.get())
      << "This FunctionNode with name `" << GetOpName() << "` has been released.";
  if (!IsReadyToRun(out_grads_)) { return false; }
  JUST(InitEmptyTensorArgs2ZerosTensor(*outputs_, out_grads_));
  TensorTuple input_grads(in_grads_.size());
  TensorTuple output_grads(out_grads_.size());
  for (int i = 0; i < out_grads_.size(); ++i) {
    output_grads.at(i) = JUST(out_grads_.at(i)->GetAccTensor());
  }
  JUST((*backward_fn_)(output_grads, &input_grads, create_graph));
  for (int i = 0; i < in_grads_.size(); ++i) {
    JUST(in_grads_.at(i)->PushPartialTensor(input_grads.at(i)));
  }
  return true;
}

void StackAutogradEngine::ClearEngine() {
  for (const auto& weak_func_node : node_list_) {
    const auto& func_node = weak_func_node.lock();
    if (func_node) { func_node->ReleaseData(); }
  }
  node_list_.clear();
}

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
    JUST(outputs.at(i)->now_grad_arg()->PushPartialTensor(out_grads.at(i)));
  }
  // Runs each FunctionNode
  for (const auto& weak_func_node : node_list_) {
    const auto& func_node = weak_func_node.lock();
    if (!func_node) { continue; }
    // CHECK_NOTNULL_OR_RETURN(func_node);
    if (JUST(func_node->Apply(create_graph))) {
      JUST(func_node->AccGrad4LeafTensor(create_graph));
      JUST(func_node->AccGrad4RetainGradTensor());
      func_node->ReleaseOutTensorArgs();
    }
  }
  if (!retain_graph) { ClearEngine(); }
  return Maybe<void>::Ok();
}

Maybe<TensorTuple> StackAutogradEngine::RunBackwardAndReturnInputsTensorGrad(
    const TensorTuple& outputs, const TensorTuple& inputs, const TensorTuple& out_grads,
    bool retain_graph, bool create_graph) {
  ClearReleasedFunctionNodes();
  std::shared_ptr<TensorTuple> input_now_grads = std::make_shared<TensorTuple>(inputs.size());
  std::vector<bool> ori_retain_grad(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    ori_retain_grad.at(i) = inputs.at(i)->retain_grad();
    inputs.at(i)->set_retain_grad(true);
  }
  for (int i = 0; i < outputs.size(); ++i) {
    JUST(outputs.at(i)->now_grad_arg()->PushPartialTensor(out_grads.at(i)));
  }
  // Runs each FunctionNode
  for (const auto& weak_func_node : node_list_) {
    const auto& func_node = weak_func_node.lock();
    if (!func_node) { continue; }
    // CHECK_NOTNULL_OR_RETURN(func_node);
    if (JUST(func_node->Apply(create_graph))) {
      JUST(func_node->AccGrad4RetainGradTensor());
      func_node->ReleaseOutTensorArgs();
    }
  }
  for (int i = 0; i < inputs.size(); ++i) {
    input_now_grads->at(i) = inputs.at(i)->acc_grad();
    if (!ori_retain_grad.at(i)) {
      inputs.at(i)->mut_acc_grad().reset();
      inputs.at(i)->set_retain_grad(false);
    }
  }
  if (!retain_graph) { ClearEngine(); }
  return input_now_grads;
}

std::shared_ptr<FunctionNode> StackAutogradEngine::AddBackwardFuncPtr(
    const std::shared_ptr<const std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>&
        backward_fn,
    const TensorTuple& inputs, TensorTuple* outputs) {
  // Firstly push function_node of tensor in stack which is leaf and requires_grad
  for (const std::shared_ptr<Tensor>& in_tensor : inputs) {
    if (in_tensor->is_leaf() && in_tensor->requires_grad()) {
      if (!in_tensor->grad_fn_node()) { AddAccumulateFunctionNode(in_tensor); }
      StackFunctionNode* stack_function_node =
          dynamic_cast<StackFunctionNode*>(in_tensor->mut_grad_fn_node().get());
      if (!stack_function_node->is_in_stack()) {
        stack_function_node->set_is_in_stack(true);
        node_list_.push_front(in_tensor->mut_grad_fn_node());
      }
    }
  }

  std::shared_ptr<StackFunctionNode> func_node =
      std::make_shared<StackFunctionNode>(backward_fn, inputs, *outputs);
  for (const std::shared_ptr<Tensor>& out_tensor : *outputs) {
    out_tensor->set_grad_fn_node(func_node);
  }
  func_node->set_is_in_stack(true);
  node_list_.push_front(func_node);
  return func_node;
}

AutogradEngine* GetThreadLocalAutogradEngine() {
  thread_local static StackAutogradEngine autograd_engine;
  return &autograd_engine;
}

Maybe<void> AddAccumulateFunctionNode(const std::shared_ptr<Tensor>& tensor) {
  auto backward_fn =
      std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
          [=](const TensorTuple& out_grads, TensorTuple* in_grads,
              bool create_graph) -> Maybe<void> { return Maybe<void>::Ok(); });
  tensor->set_grad_fn_node(
      std::make_shared<StackFunctionNode>(backward_fn, TensorTuple(), TensorTuple({tensor})));
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
