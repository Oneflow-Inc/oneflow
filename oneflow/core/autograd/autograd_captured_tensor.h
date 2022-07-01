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
#ifndef ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_CAPTURED_TENSOR_H_
#define ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_CAPTURED_TENSOR_H_

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

class AutogradCapturedTensor final : public ProxyTensor<AutogradCapturedTensor> {
 public:
  static Maybe<AutogradCapturedTensor> MakeTensor(const std::shared_ptr<Tensor>& tensor) {
    if (tensor->requires_grad()) {
      CHECK_NOTNULL_OR_RETURN(tensor->grad_fn_node().get())
          << Error::RuntimeError()
          << "a grad function node is expected for the captured tensor "
             "which requires_grad is True.";
    }
    std::shared_ptr<AutogradCapturedTensor> captured_tensor(
        new AutogradCapturedTensor(JUST(tensor->detach())));
    captured_tensor->set_autograd_meta(tensor->mut_autograd_meta());
    captured_tensor->grad_fn_node_ = tensor->mut_grad_fn_node();
    return captured_tensor;
  }

  std::shared_ptr<const FunctionNode> grad_fn_node() const override { return grad_fn_node_.lock(); }
  void set_grad_fn_node(const std::shared_ptr<FunctionNode>& grad_fn_node) override {
    PRINT_BUG_PROMPT_AND_ABORT();
  }
  std::shared_ptr<FunctionNode> mut_grad_fn_node() override { return grad_fn_node_.lock(); }

  std::shared_ptr<Tensor> contiguous() const override {
    const auto& tensor = std::const_pointer_cast<Tensor>(shared_from_this());
    if (tensor_->is_contiguous()) { return tensor; }
    return CHECK_JUST(functional::ToContiguous(tensor));
  }

 private:
  explicit AutogradCapturedTensor(const std::shared_ptr<Tensor>& tensor)
      : ProxyTensor<AutogradCapturedTensor>(tensor) {}

 private:
  std::weak_ptr<FunctionNode> grad_fn_node_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_CAPTURED_TENSOR_H_
