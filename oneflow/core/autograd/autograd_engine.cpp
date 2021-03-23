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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr_helper.h"

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
  const auto& zero_like = JUST(helper::ZeroLikeOp());
  for (int i = 0; i < out_grads.size(); ++i) {
    if (out_grads.at(i)->Empty()) {
      TensorTuple output(1);
      GetInterpreter()->Apply(zero_like, {outputs.at(i)}, output);
      out_grads.at(i)->PushPartialTensor(output.at(0));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> CopyOrAccGrad(Tensor& tensor, bool autograd_mode) {
  autograd::AutogradMode mode(autograd_mode);
  const auto& tensor_arg = tensor.now_grad_arg();
  if (tensor.acc_grad()) {
    TensorTuple input = {tensor.acc_grad(), tensor_arg->GetAccTensor().GetPtrOrThrow()};
    TensorTuple output(1);
    const auto& add = JUST(helper::AddOp());
    GetInterpreter()->Apply(add, input, output);
    tensor.set_acc_grad(output.at(0));
  } else {
    tensor.set_acc_grad(tensor_arg->GetAccTensor().GetPtrOrThrow());
  }
  return Maybe<void>::Ok();
}

}  // namespace

StackFunctionNode::StackFunctionNode(
    const std::shared_ptr<const std::function<Maybe<void>()>>& backward_fn,
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
}

Maybe<void> StackFunctionNode::AccGrad4RetainGradTensor() {
  for (int i = 0; i < outputs_->size(); ++i) {
    if (outputs_->at(i)->retain_grad() && outputs_->at(i)->requires_grad()) {
      JUST(CopyOrAccGrad(outputs_->at(i), /*autograd_mode=*/false));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> StackFunctionNode::AccGrad4LeafTensor(bool create_graph) {
  for (int i = 0; i < outputs_->size(); ++i) {
    if (outputs_->at(i)->is_leaf() && outputs_->at(i)->requires_grad()) {
      JUST(CopyOrAccGrad(outputs_->at(i), /*autograd_mode=*/create_graph));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> StackFunctionNode::GetNowGrad(TensorTuple* input_now_grads,
                                          const HashMap<TensorArg*, size_t>& tensor_arg2idx) const {
  for (const auto& out_tensor : *outputs_) {
    const auto& iter = tensor_arg2idx.find(out_tensor->now_grad_arg().get());
    if (iter != tensor_arg2idx.end()) {
      input_now_grads->at(iter->second) = out_tensor->now_grad_arg()->GetAccTensor()->detach();
    }
  }
  return Maybe<void>::Ok();
}

void StackFunctionNode::ReleaseOutTensorArgs() {
  for (const std::shared_ptr<TensorArg>& tensor_arg : out_grads_) { tensor_arg->Release(); }
}

void StackFunctionNode::ReleaseData() {
  inputs_.reset();
  outputs_.reset();
  in_grads_.clear();
  out_grads_.clear();
  backward_fn_.reset();
}

Maybe<void> StackFunctionNode::Apply(bool create_graph) {
  CHECK_OR_RETURN(!backward_fn_) << "This FunctionNode with name `" << GetOpName()
                                 << "` has been released.";
  if (!IsReadyToRun(out_grads_)) { return Maybe<void>::Ok(); }
  InitEmptyTensorArgs2ZerosTensor(*outputs_, out_grads_);
  TODO();  // wangyinggang: Calls backward_fn_ and passes arguments according to AutogradInterpreter
  return Maybe<void>::Ok();
}

void StackAutogradEngine::ClearEngine() {
  for (const auto& weak_func_node : node_list_) {
    const auto& func_node = weak_func_node.lock();
    if (func_node) { func_node->ReleaseData(); }
  }
  node_list_.clear();
}

Maybe<void> StackAutogradEngine::RunBackwardAndSaveGrads4LeafTensor(const TensorTuple& outputs,
                                                                    const TensorTuple& out_grads,
                                                                    bool retain_graph,
                                                                    bool create_graph) {
  for (int i = 0; i < outputs.size(); ++i) {
    outputs.at(i)->now_grad_arg()->PushPartialTensor(out_grads.at(i));
  }
  // Runs each FunctionNode
  for (const auto& weak_func_node : node_list_) {
    const auto& func_node = weak_func_node.lock();
    if (!func_node) { continue; }
    JUST(func_node->Apply(create_graph));
    JUST(func_node->AccGrad4LeafTensor());
    JUST(func_node->AccGrad4RetainGradTensor());
    func_node->ReleaseOutTensorArgs();
  }
  if (!retain_graph) { ClearEngine(); }
  return Maybe<void>::Ok();
}

Maybe<TensorTuple> StackAutogradEngine::RunBackwardAndReturnInputsTensorGrad(
    const TensorTuple& outputs, const TensorTuple& inputs, const TensorTuple& out_grads,
    bool retain_graph, bool create_graph) {
  std::shared_ptr<TensorTuple> input_now_grads = std::make_shared<TensorTuple>(inputs.size());
  HashMap<TensorArg*, size_t> tensor_arg2idx;
  for (int i = 0; i < inputs.size(); ++i) {
    tensor_arg2idx.emplace(inputs.at(i)->now_grad_arg(), i);
  }
  for (int i = 0; i < outputs.size(); ++i) {
    outputs.at(i)->now_grad_arg()->PushPartialTensor(out_grads.at(i));
  }
  // Runs each FunctionNode
  for (const auto& weak_func_node : node_list_) {
    const auto& func_node = weak_func_node.lock();
    if (!func_node) { continue; }
    JUST(func_node->Apply(create_graph));
    JUST(func_node->GetNowGrad(input_now_grads.get(), tensor_arg2idx));
    JUST(func_node->AccGrad4RetainGradTensor());
    func_node->ReleaseOutTensorArgs();
  }
  if (!retain_graph) { ClearEngine(); }
  return input_now_grads;
}

std::shared_ptr<FunctionNode> StackAutogradEngine::AddBackwardFuncPtr(
    const std::shared_ptr<const std::function<Maybe<void>()>>& backward_fn,
    const TensorTuple& inputs, TensorTuple* outputs) {
  std::shared_ptr<FunctionNode> func_node =
      std::make_shared<StackFunctionNode>(backward_fn, inputs, *outputs);
  for (const std::shared_ptr<Tensor>& out_tensor : *outputs) {
    out_tensor->set_grad_fn_node(func_node);
  }
  node_list_.push_front(func_node);
  return func_node;
}

AutogradEngine* GetThreadLocalAutogradEngine() {
  thread_local static StackAutogradEngine autograd_engine;
  return &autograd_engine;
}

}  // namespace one
}  // namespace oneflow
