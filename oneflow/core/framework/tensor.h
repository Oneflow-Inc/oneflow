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
#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.cfg.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/framework/rpc_token.h"
#include "oneflow/core/common/error.h"

namespace oneflow {

namespace cfg {
class ParallelDistribution;
}
class Device;

namespace one {

class FunctionNode;

class ConsistentTensor;
class MirroredTensor;

class Tensor {
 public:
  virtual ~Tensor() = default;

  // Getters
  virtual int64_t nelement() const = 0;
  virtual int64_t dim(int64_t index) const = 0;

  virtual const std::shared_ptr<const Shape>& shape() const = 0;
  virtual DataType dtype() const = 0;
  virtual Maybe<RpcToken> rpc_token() const = 0;
  virtual Maybe<Symbol<cfg::ParallelDistribution>> parallel_distribution() const = 0;
  virtual Maybe<Symbol<ParallelDesc>> parallel_desc() const = 0;
  virtual Maybe<Symbol<Device>> device() const = 0;
  virtual Maybe<Symbol<Device>*> mut_device() = 0;
  virtual int64_t ndim() const = 0;
  virtual bool is_cuda() const = 0;
  virtual bool is_consistent() const = 0;
  virtual bool is_local() const { return !is_consistent(); }
  virtual bool is_lazy() const = 0;
  virtual bool is_eager() const { return !is_lazy(); }
  virtual const TensorMeta& tensor_meta() const = 0;
  virtual Maybe<Symbol<ConsistentTensorMeta>> consistent_tensor_meta() const { OF_UNIMPLEMENTED(); }

  // Getters valid only for EagerMirroredTensor
  virtual Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() { OF_UNIMPLEMENTED(); }
  virtual Maybe<vm::EagerBlobObject> eager_blob_object() const = 0;
  virtual Maybe<VmLocalDepObject> compute_local_dep_object() const = 0;
  virtual Maybe<bool> has_eager_blob_object() const = 0;
  virtual Maybe<TensorStorage> tensor_storage() const { OF_UNIMPLEMENTED(); }
  virtual Maybe<const Stride> stride() const { OF_UNIMPLEMENTED(); }
  virtual Maybe<int64_t> storage_offset() const { OF_UNIMPLEMENTED(); }

  // Getters/Setters valid only for EagerConsistentTensor
  virtual Maybe<const Optional<Symbol<cfg::ParallelDistribution>>&>
  consumer_parallel_distribution_constraint() const {
    OF_UNIMPLEMENTED();
  }
  virtual Maybe<MirroredTensor> cur_rank_phy_tensor() const { OF_UNIMPLEMENTED(); }
  virtual Maybe<void> set_consumer_parallel_distribution_constraint(
      Symbol<cfg::ParallelDistribution> val) {
    OF_UNIMPLEMENTED();
  }

  // Getters for autograd
  virtual bool requires_grad() const = 0;
  virtual bool is_leaf() const = 0;
  virtual bool retain_grad() const = 0;
  virtual std::shared_ptr<const FunctionNode> grad_fn_node() const = 0;
  virtual Maybe<Tensor> acc_grad() const = 0;
  virtual Maybe<TensorArg> current_grad() const = 0;
  virtual Maybe<Tensor> detach() const = 0;
  virtual Maybe<Tensor> clone() const = 0;
  virtual std::shared_ptr<Tensor> data() const = 0;

  // Setters for autograd
  virtual void set_requires_grad(bool requires_grad) = 0;
  virtual Maybe<void> set_retain_grad(bool retain_grad) = 0;
  virtual void set_grad_fn_node(const std::shared_ptr<FunctionNode>& grad_fn_node) = 0;
  virtual const std::shared_ptr<FunctionNode>& mut_grad_fn_node() = 0;
  virtual Maybe<void> set_acc_grad(const std::shared_ptr<Tensor>& grad) = 0;
  virtual Maybe<Tensor> mut_acc_grad() = 0;
  virtual void set_is_leaf(bool is_leaf) = 0;
  virtual std::shared_ptr<AutogradMeta> mut_autograd_meta() = 0;
  virtual bool has_autograd_meta() const = 0;
  virtual void set_autograd_meta(const std::shared_ptr<AutogradMeta>& autograd_meta) = 0;

  virtual user_op::TensorDesc* mut_tensor_meta() = 0;

  virtual Maybe<MirroredTensor> AsMirroredTensor() = 0;

 protected:
  Tensor() = default;
};

template<typename DerivedT>
class TensorIf : public Tensor {
 public:
  virtual ~TensorIf() = default;

  // Getters for autograd
  // acc_grad is tensor's accumulated grad in more than once backward operation,
  // and current_grad is temporary grad to shared data with different FunctionNode
  std::shared_ptr<const FunctionNode> grad_fn_node() const override { return grad_fn_node_; }

  // Setters for autograd
  void set_grad_fn_node(const std::shared_ptr<FunctionNode>& grad_fn_node) override {
    grad_fn_node_ = grad_fn_node;
  }
  const std::shared_ptr<FunctionNode>& mut_grad_fn_node() override { return grad_fn_node_; }

 protected:
  TensorIf() = default;
  std::shared_ptr<FunctionNode> grad_fn_node_;
};

class Parameter final : public TensorIf<Parameter> {
 public:
  Parameter(std::shared_ptr<Tensor> tensor, bool requires_grad) {
    while (auto parameter = std::dynamic_pointer_cast<Parameter>(tensor)) {
      tensor = parameter->tensor_;
    }
    this->tensor_ = std::move(tensor);
    // TODO: in `y = flow.nn.Parameter(x)`, y should have its own "requires_grad" field
    // (align with PyTorch) instead of sharing it with x
    this->tensor_->set_requires_grad(requires_grad);
  }

  const std::shared_ptr<const Shape>& shape() const override { return tensor_->shape(); }
  DataType dtype() const override { return tensor_->dtype(); }
  Maybe<Symbol<cfg::ParallelDistribution>> parallel_distribution() const override {
    return tensor_->parallel_distribution();
  }
  Maybe<Symbol<ParallelDesc>> parallel_desc() const override { return tensor_->parallel_desc(); }
  Maybe<Symbol<Device>> device() const override { return tensor_->device(); }
  Maybe<Symbol<Device>*> mut_device() override { return tensor_->mut_device(); }
  int64_t ndim() const override { return tensor_->ndim(); }
  bool is_cuda() const override { return tensor_->is_cuda(); }
  bool is_consistent() const override { return tensor_->is_consistent(); }
  bool is_local() const override { return tensor_->is_local(); }
  bool is_lazy() const override { return tensor_->is_lazy(); }
  bool is_eager() const override { return tensor_->is_eager(); }
  const TensorMeta& tensor_meta() const override { return tensor_->tensor_meta(); }
  Maybe<Symbol<ConsistentTensorMeta>> consistent_tensor_meta() const override {
    return tensor_->consistent_tensor_meta();
  }

  Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() override {
    return tensor_->mut_eager_mirrored_tensor_impl();
  }
  Maybe<vm::EagerBlobObject> eager_blob_object() const override {
    return tensor_->eager_blob_object();
  }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override {
    return tensor_->compute_local_dep_object();
  }
  Maybe<bool> has_eager_blob_object() const override { return tensor_->has_eager_blob_object(); }
  Maybe<TensorStorage> tensor_storage() const override { return tensor_->tensor_storage(); }
  Maybe<const Stride> stride() const override { return tensor_->stride(); }
  Maybe<int64_t> storage_offset() const override { return tensor_->storage_offset(); }

  Maybe<const Optional<Symbol<cfg::ParallelDistribution>>&>
  consumer_parallel_distribution_constraint() const override {
    return tensor_->consumer_parallel_distribution_constraint();
  }
  Maybe<RpcToken> rpc_token() const override { return tensor_->rpc_token(); }
  Maybe<MirroredTensor> cur_rank_phy_tensor() const override {
    return tensor_->cur_rank_phy_tensor();
  }
  Maybe<void> set_consumer_parallel_distribution_constraint(
      Symbol<cfg::ParallelDistribution> val) override {
    return tensor_->set_consumer_parallel_distribution_constraint(val);
  }

  bool requires_grad() const override { return tensor_->requires_grad(); }
  bool is_leaf() const override { return true; }
  bool retain_grad() const override { return tensor_->retain_grad(); }
  Maybe<Tensor> acc_grad() const override { return tensor_->acc_grad(); }
  Maybe<TensorArg> current_grad() const override { return tensor_->current_grad(); }
  Maybe<Tensor> detach() const override { return tensor_->detach(); }
  Maybe<Tensor> clone() const override { return tensor_->clone(); }
  std::shared_ptr<Tensor> data() const override { return tensor_->data(); }

  void set_requires_grad(bool requires_grad) override {
    return tensor_->set_requires_grad(requires_grad);
  }
  Maybe<void> set_retain_grad(bool retain_grad) override {
    return tensor_->set_retain_grad(retain_grad);
  }
  Maybe<void> set_acc_grad(const std::shared_ptr<Tensor>& grad) override {
    return tensor_->set_acc_grad(grad);
  }
  Maybe<Tensor> mut_acc_grad() override { return tensor_->mut_acc_grad(); }
  void set_is_leaf(bool is_leaf) override { return tensor_->set_is_leaf(is_leaf); }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() override {
    return tensor_->mut_autograd_meta();
  }
  bool has_autograd_meta() const override { return tensor_->has_autograd_meta(); }
  void set_autograd_meta(const std::shared_ptr<AutogradMeta>& autograd_meta) override {
    return tensor_->set_autograd_meta(autograd_meta);
  }

  user_op::TensorDesc* mut_tensor_meta() override { return tensor_->mut_tensor_meta(); }

  Maybe<MirroredTensor> AsMirroredTensor() override {
    if (const auto& mirrored_tensor = std::dynamic_pointer_cast<MirroredTensor>(tensor_)) {
      return mirrored_tensor;
    }
    UNIMPLEMENTED_THEN_RETURN();
  }

  int64_t nelement() const override { return tensor_->nelement(); }
  int64_t dim(int64_t index) const override { return tensor_->dim(index); }

 private:
  std::shared_ptr<Tensor> tensor_;
};

class MirroredTensor final : public TensorIf<MirroredTensor>,
                             public std::enable_shared_from_this<MirroredTensor> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MirroredTensor);
  MirroredTensor() = default;
  explicit MirroredTensor(const std::shared_ptr<MirroredTensorImpl>& impl) { impl_ = impl; }
  ~MirroredTensor() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return impl_->shape(); }
  DataType dtype() const override { return impl_->dtype(); }
  Maybe<RpcToken> rpc_token() const override { OF_UNIMPLEMENTED(); }
  Maybe<Symbol<cfg::ParallelDistribution>> parallel_distribution() const override {
    OF_UNIMPLEMENTED();
  }
  Maybe<Symbol<ParallelDesc>> parallel_desc() const override { OF_UNIMPLEMENTED(); }
  Maybe<Symbol<Device>> device() const override { return impl_->device(); }
  Maybe<Symbol<Device>*> mut_device() override { return impl_->mut_device(); }
  bool is_lazy() const override { return impl_->is_lazy(); }
  bool is_consistent() const override { return false; }
  int64_t ndim() const override;
  bool is_cuda() const override;
  int64_t dim(int64_t index) const override;
  int64_t nelement() const override;
  std::shared_ptr<Tensor> data() const override;
  const TensorMeta& tensor_meta() const override { return *impl_->tensor_meta(); }

  // Getters valid only for EagerMirroredTensor
  Maybe<vm::EagerBlobObject> eager_blob_object() const override {
    return impl_->eager_blob_object();
  }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override {
    return impl_->compute_local_dep_object();
  }
  Maybe<TensorStorage> tensor_storage() const override { return impl_->tensor_storage(); }
  Maybe<bool> has_eager_blob_object() const override { return impl_->has_eager_blob_object(); }
  Maybe<const Stride> stride() const override { return impl_->stride(); }
  Maybe<int64_t> storage_offset() const override { return impl_->storage_offset(); }

  // Getters for autograd
  Maybe<Tensor> acc_grad() const override { return impl_->acc_grad(); }
  Maybe<TensorArg> current_grad() const override { return impl_->current_grad(); }
  bool requires_grad() const override { return impl_->requires_grad(); }
  bool is_leaf() const override { return impl_->is_leaf(); }
  bool retain_grad() const override { return impl_->retain_grad(); }
  bool has_autograd_meta() const override { return impl_->has_autograd_meta(); }

  // Setters for autograd
  Maybe<void> set_acc_grad(const std::shared_ptr<Tensor>& grad) override {
    return impl_->set_acc_grad(grad);
  }
  void set_requires_grad(bool requires_grad) override { impl_->set_requires_grad(requires_grad); }
  Maybe<void> set_retain_grad(bool retain_grad) override {
    return impl_->set_retain_grad(retain_grad);
  }
  Maybe<Tensor> mut_acc_grad() override { return impl_->mut_acc_grad(); }
  void set_is_leaf(bool is_leaf) override { impl_->set_is_leaf(is_leaf); }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() override { return impl_->mut_autograd_meta(); }
  void set_autograd_meta(const std::shared_ptr<AutogradMeta>& autograd_meta) override {
    impl_->set_autograd_meta(autograd_meta);
  }

  // Operators for tensor
  Maybe<Tensor> detach() const override;
  Maybe<Tensor> clone() const override;

  static Maybe<MirroredTensor> MakeTensor(const std::shared_ptr<const Shape>& shape, DataType dtype,
                                          const Symbol<Device>& device, bool is_lazy,
                                          bool requires_grad, bool is_leaf);
  MirroredTensorImpl* mut_impl() { return impl_.get(); }
  Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() override {
    return impl_->mut_eager_mirrored_tensor_impl();
  }
  user_op::TensorDesc* mut_tensor_meta() override { return impl_->mut_tensor_meta(); }

  Maybe<MirroredTensor> MakeEagerTensor(
      const std::shared_ptr<vm::EagerBlobObject> eager_blob_object, const Symbol<Device>& device,
      const std::shared_ptr<TensorStorage> tensor_storage, bool requires_grad, bool is_leaf);

  Maybe<MirroredTensor> AsMirroredTensor() override { return shared_from_this(); }

 private:
  std::shared_ptr<MirroredTensorImpl> impl_;
};

class ConsistentTensor final : public TensorIf<ConsistentTensor> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConsistentTensor);
  ConsistentTensor() = default;
  explicit ConsistentTensor(const std::shared_ptr<ConsistentTensorImpl>& impl) { impl_ = impl; }
  ~ConsistentTensor() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return impl_->shape(); }
  DataType dtype() const override { return impl_->dtype(); }
  Maybe<RpcToken> rpc_token() const override { return impl_->rpc_token(); }
  Maybe<Symbol<cfg::ParallelDistribution>> parallel_distribution() const override {
    return impl_->parallel_distribution();
  }
  Maybe<Symbol<ParallelDesc>> parallel_desc() const override { return impl_->parallel_desc(); }
  Maybe<Symbol<Device>> device() const override { OF_UNIMPLEMENTED(); }
  Maybe<Symbol<Device>*> mut_device() override { OF_UNIMPLEMENTED(); }
  bool is_lazy() const override { return impl_->is_lazy(); }
  bool is_consistent() const override { return true; }
  Maybe<const Optional<Symbol<cfg::ParallelDistribution>>&>
  consumer_parallel_distribution_constraint() const override {
    return impl_->consumer_parallel_distribution_constraint();
  }
  Maybe<MirroredTensor> cur_rank_phy_tensor() const override {
    return impl_->cur_rank_phy_tensor();
  }
  int64_t ndim() const override;
  bool is_cuda() const override;
  int64_t dim(int64_t index) const override;
  int64_t nelement() const override;
  std::shared_ptr<Tensor> data() const override;

  // Getters valid only for EagerMirroredTensor
  Maybe<vm::EagerBlobObject> eager_blob_object() const override {
    return impl_->eager_blob_object();
  }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override {
    return impl_->compute_local_dep_object();
  }
  const TensorMeta& tensor_meta() const override { return *impl_->tensor_meta(); }
  Maybe<TensorStorage> tensor_storage() const override { return impl_->tensor_storage(); }
  Maybe<bool> has_eager_blob_object() const override { return impl_->has_eager_blob_object(); }

  // Setters
  Maybe<void> set_consumer_parallel_distribution_constraint(
      Symbol<cfg::ParallelDistribution> val) override {
    impl_->set_consumer_parallel_distribution_constraint(val);
    return Maybe<void>::Ok();
  }

  // Getters for autograd
  Maybe<Tensor> acc_grad() const override { return impl_->acc_grad(); }
  Maybe<TensorArg> current_grad() const override { return impl_->current_grad(); }
  bool requires_grad() const override { return impl_->requires_grad(); }
  bool is_leaf() const override { return impl_->is_leaf(); }
  bool retain_grad() const override { return impl_->retain_grad(); }
  bool has_autograd_meta() const override { return impl_->has_autograd_meta(); }

  // Setters for autograd
  Maybe<void> set_acc_grad(const std::shared_ptr<Tensor>& grad) override {
    return impl_->set_acc_grad(grad);
  }
  Maybe<Tensor> mut_acc_grad() override { return impl_->mut_acc_grad(); }
  void set_requires_grad(bool requires_grad) override { impl_->set_requires_grad(requires_grad); }
  Maybe<void> set_retain_grad(bool retain_grad) override {
    return impl_->set_retain_grad(retain_grad);
  }
  void set_is_leaf(bool is_leaf) override { impl_->set_is_leaf(is_leaf); }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() override { return impl_->mut_autograd_meta(); }
  void set_autograd_meta(const std::shared_ptr<AutogradMeta>& autograd_meta) override {
    impl_->set_autograd_meta(autograd_meta);
  }

  // Operators for tensor
  Maybe<Tensor> detach() const override;
  Maybe<Tensor> clone() const override { return Error::Unimplemented(); }

  static Maybe<ConsistentTensor> MakeTensor(const std::shared_ptr<const Shape>& shape,
                                            DataType dtype,
                                            Symbol<cfg::ParallelDistribution> parallel_distribution,
                                            Symbol<ParallelDesc> parallel_desc, bool is_lazy,
                                            bool requires_grad, bool is_leaf);

  ConsistentTensorImpl* mut_impl() { return impl_.get(); }

  Maybe<Symbol<ConsistentTensorMeta>> consistent_tensor_meta() const override {
    return impl_->tensor_meta();
  }

  user_op::TensorDesc* mut_tensor_meta() override { return impl_->mut_tensor_meta(); }

  Maybe<MirroredTensor> AsMirroredTensor() override { UNIMPLEMENTED_THEN_RETURN(); }

 private:
  std::shared_ptr<ConsistentTensorImpl> impl_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
