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
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"

namespace oneflow {

class Shape;

class Device;
class ParallelDesc;
class NdSbp;

namespace one {

class Tensor;
class TensorArg;
class LocalTensor;

class AutogradMeta final {
 public:
  AutogradMeta() = delete;
  AutogradMeta(bool requires_grad, bool is_leaf);

  // Getters
  const std::shared_ptr<Tensor>& acc_grad() const { return acc_grad_; }
  const std::shared_ptr<TensorArg>& current_grad() const { return current_grad_; }
  // get current grad processed by hooks
  Maybe<Tensor> current_grad_value() const;
  bool requires_grad() const { return requires_grad_; }
  bool is_leaf() const { return is_leaf_; }
  bool retain_grad() const { return retain_grad_; }
  using Hook = std::function<std::shared_ptr<Tensor>(const std::shared_ptr<const Tensor>&)>;
  const std::vector<Hook>& hooks() const { return hooks_; }
  const std::vector<Hook>& post_grad_accumulation_hooks() const {
    return post_grad_accumulation_hooks_;
  }

  // Setters
  Maybe<void> set_acc_grad(const std::shared_ptr<Tensor>& grad);
  std::shared_ptr<Tensor> mut_acc_grad() { return acc_grad_; }
  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
  void set_retain_grad(bool retain_grad) { retain_grad_ = retain_grad; }
  void set_is_leaf(bool is_leaf) { is_leaf_ = is_leaf; }
  void add_hook(const Hook& hook) { hooks_.emplace_back(hook); }
  void add_post_grad_accumulation_hook(const Hook& hook) {
    post_grad_accumulation_hooks_.emplace_back(hook);
  }

 private:
  bool is_leaf_;

  // Only meaningful on leaf Tensors (must be false otherwise)
  bool requires_grad_;

  // Only meaningful on non_leaf Tensors (must be false otherwise)
  bool retain_grad_;

  std::shared_ptr<Tensor> acc_grad_;
  std::shared_ptr<TensorArg> current_grad_;
  std::vector<Hook> hooks_;
  std::vector<Hook> post_grad_accumulation_hooks_;
};

inline std::shared_ptr<AutogradMeta> NewAutogradMeta(bool requires_grad, bool is_leaf) {
  return std::shared_ptr<AutogradMeta>(new AutogradMeta(requires_grad, is_leaf));
}

class TensorInfo final {
 public:
  TensorInfo() = delete;
  explicit TensorInfo(const Tensor& tensor);

  Maybe<Tensor> zeros() const;
  Optional<Symbol<ParallelDesc>> placement() const { return parallel_desc_; }
  Optional<Symbol<NdSbp>> sbp() const { return nd_sbp_; }

 private:
  std::shared_ptr<const Shape> shape_;
  Symbol<DType> dtype_;
  Optional<Symbol<Device>> device_;               // for local tensor
  Optional<Symbol<ParallelDesc>> parallel_desc_;  // for global tensor
  Optional<Symbol<NdSbp>> nd_sbp_;                // for global tensor
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_META_H_
