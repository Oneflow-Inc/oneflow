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

#ifndef ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_META_H_
#define ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_META_H_

#include <memory>
#include "oneflow/core/framework/tensor_arg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class Shape;
class DType;

namespace one {

class Tensor;
class TensorArg;

class AutogradMeta final {
 public:
  AutogradMeta() = delete;
  AutogradMeta(bool requires_grad, bool is_leaf)
      : is_leaf_(is_leaf),
        requires_grad_(requires_grad),
        retain_grad_(false),
        now_grad_arg_(new TensorArg) {}

  // Getters
  const std::shared_ptr<Tensor>& acc_grad() const { return acc_grad_; }
  const std::shared_ptr<TensorArg>& now_grad_arg() const { return now_grad_arg_; }
  bool requires_grad() const { return requires_grad_; }
  bool is_leaf() const { return is_leaf_; }
  bool retain_grad() const { return retain_grad_; }

  // Setters
  void set_acc_grad(const std::shared_ptr<Tensor>& grad) { acc_grad_ = grad; }
  std::shared_ptr<Tensor> mut_acc_grad() { return acc_grad_; }
  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
  void set_retain_grad(bool retain_grad) { retain_grad_ = retain_grad; }
  void set_is_leaf(bool is_leaf) { is_leaf_ = is_leaf; }

 private:
  bool is_leaf_;

  // Only meaningful on leaf Tensors (must be false otherwise)
  bool requires_grad_;

  // Oney meaningful on non_leaf Tensors (must be false otherwise)
  bool retain_grad_;

  std::shared_ptr<Tensor> acc_grad_;
  std::shared_ptr<TensorArg> now_grad_arg_;
};

inline std::shared_ptr<AutogradMeta> NewAutogradMeta(bool requires_grad, bool is_leaf) {
  return std::shared_ptr<AutogradMeta>(new AutogradMeta(requires_grad, is_leaf));
}

class TensorInfo final {
 public:
  TensorInfo() = delete;
  explicit TensorInfo(const Tensor& tensor);

  Maybe<Tensor> zeros() const;

 private:
  std::shared_ptr<const Shape> shape_;
  std::shared_ptr<const DType> dtype_;
  // TODO: Add device info
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_META_H_
