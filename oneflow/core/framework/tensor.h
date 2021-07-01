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
  virtual const std::shared_ptr<const Shape>& shape() const = 0;
  virtual DataType dtype() const = 0;
  virtual Maybe<Symbol<cfg::ParallelDistribution>> parallel_distribution() const = 0;
  virtual Maybe<Symbol<ParallelDesc>> parallel_desc() const = 0;
  virtual Maybe<Symbol<Device>> device() const = 0;
  virtual Maybe<Symbol<Device>*> mut_device() { OF_UNIMPLEMENTED(); }
  virtual bool is_consistent() const = 0;
  virtual bool is_lazy() const = 0;
  virtual const TensorMeta& tensor_meta() const = 0;
  virtual Maybe<Symbol<ConsistentTensorMeta>> consistent_tensor_meta() const { OF_UNIMPLEMENTED(); }

  // Getters valid only for EagerMirroredTensor
  virtual Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() { OF_UNIMPLEMENTED(); }
  virtual Maybe<vm::EagerBlobObject> eager_blob_object() const = 0;
  virtual Maybe<VmLocalDepObject> compute_local_dep_object() const = 0;
  virtual Maybe<TensorStorage> tensor_storage() const { OF_UNIMPLEMENTED(); }

  // Getters/Setters valid only for EagerConsistentTensor
  virtual Maybe<Symbol<cfg::ParallelDistribution>> consumer_parallel_distribution_constraint()
      const {
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
  virtual const std::shared_ptr<Tensor>& acc_grad() const = 0;
  virtual const std::shared_ptr<TensorArg>& now_grad_arg() const = 0;
  virtual Maybe<Tensor> detach() const = 0;

  // Setters for autograd
  virtual void set_requires_grad(bool requires_grad) = 0;
  virtual void set_retain_grad(bool retain_grad) = 0;
  virtual void set_grad_fn_node(const std::shared_ptr<FunctionNode>& grad_fn_node) = 0;
  virtual const std::shared_ptr<FunctionNode>& mut_grad_fn_node() = 0;
  virtual void set_acc_grad(const std::shared_ptr<Tensor>& grad) = 0;
  virtual std::shared_ptr<Tensor> mut_acc_grad() = 0;
  virtual void set_is_leaf(bool is_leaf) = 0;
  virtual std::shared_ptr<AutogradMeta> mut_autograd_meta() = 0;

  virtual user_op::TensorDesc* mut_tensor_meta() = 0;

 protected:
  Tensor() = default;
};

template<typename DerivedT>
class TensorIf : public Tensor, public std::enable_shared_from_this<TensorIf<DerivedT>> {
 public:
  virtual ~TensorIf() = default;

  // Getters
  virtual int64_t ndim() const = 0;
  virtual bool is_cuda() const = 0;
  virtual int64_t nelement() const = 0;
  virtual int64_t dim(int64_t index) const = 0;

  // Getters for autograd
  // acc_grad is tensor's accumulated grad in more than once backward operation,
  // and now_grad_arg is temporary grad to shared data with different FunctionNode
  std::shared_ptr<const FunctionNode> grad_fn_node() const override { return grad_fn_node_; }
  // used by pybind11 only
  Maybe<DerivedT> api_acc_grad() const {
    const std::shared_ptr<Tensor>& tensor = acc_grad();
    return cast_for_api(tensor);
  }

  // Setters for autograd
  void set_grad_fn_node(const std::shared_ptr<FunctionNode>& grad_fn_node) override {
    grad_fn_node_ = grad_fn_node;
  }
  const std::shared_ptr<FunctionNode>& mut_grad_fn_node() override { return grad_fn_node_; }

  Maybe<Tensor> detach() const override {
    return std::static_pointer_cast<Tensor>(JUST(api_detach()));
  }

  // Operators for tensor
  // used by pybind11 only
  virtual Maybe<DerivedT> api_detach() const = 0;

 protected:
  TensorIf() = default;
  std::shared_ptr<FunctionNode> grad_fn_node_;

 private:
  Maybe<DerivedT> cast_for_api(const std::shared_ptr<Tensor>& tensor) const {
    if (!tensor) { return std::shared_ptr<DerivedT>(); }
    const auto& ptr = std::dynamic_pointer_cast<DerivedT>(tensor);
    CHECK_OR_RETURN(ptr) << Error::ValueError("Tensor Cast Error");
    return ptr;
  }
};

class MirroredTensor final : public TensorIf<MirroredTensor> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MirroredTensor);
  MirroredTensor() = default;
  explicit MirroredTensor(const std::shared_ptr<MirroredTensorImpl>& impl) { impl_ = impl; }
  ~MirroredTensor() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return impl_->shape(); }
  DataType dtype() const override { return impl_->dtype(); }
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
  std::shared_ptr<MirroredTensor> data() const;
  const TensorMeta& tensor_meta() const override { return *impl_->tensor_meta(); }

  // Getters valid only for EagerMirroredTensor
  Maybe<vm::EagerBlobObject> eager_blob_object() const override {
    return impl_->eager_blob_object();
  }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override {
    return impl_->compute_local_dep_object();
  }
  Maybe<TensorStorage> tensor_storage() const override { return impl_->tensor_storage(); }

  // Getters for autograd
  const std::shared_ptr<Tensor>& acc_grad() const override { return impl_->acc_grad(); }
  const std::shared_ptr<TensorArg>& now_grad_arg() const override { return impl_->now_grad_arg(); }
  bool requires_grad() const override { return impl_->requires_grad(); }
  bool is_leaf() const override { return impl_->is_leaf(); }
  bool retain_grad() const override { return impl_->retain_grad(); }

  // Setters for autograd
  void set_acc_grad(const std::shared_ptr<Tensor>& grad) override { impl_->set_acc_grad(grad); }
  void set_requires_grad(bool requires_grad) override { impl_->set_requires_grad(requires_grad); }
  void set_retain_grad(bool retain_grad) override { impl_->set_retain_grad(retain_grad); }
  std::shared_ptr<Tensor> mut_acc_grad() override { return impl_->mut_acc_grad(); }
  void set_is_leaf(bool is_leaf) override { impl_->set_is_leaf(is_leaf); }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() override { return impl_->mut_autograd_meta(); }

  // Operators for tensor
  Maybe<MirroredTensor> api_detach() const override;

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
  Maybe<Symbol<cfg::ParallelDistribution>> parallel_distribution() const override {
    return impl_->parallel_distribution();
  }
  Maybe<Symbol<ParallelDesc>> parallel_desc() const override { return impl_->parallel_desc(); }
  Maybe<Symbol<Device>> device() const override { OF_UNIMPLEMENTED(); }
  bool is_lazy() const override { return impl_->is_lazy(); }
  bool is_consistent() const override { return true; }
  Maybe<Symbol<cfg::ParallelDistribution>> consumer_parallel_distribution_constraint()
      const override {
    return impl_->consumer_parallel_distribution_constraint();
  }
  Maybe<MirroredTensor> cur_rank_phy_tensor() const { return impl_->cur_rank_phy_tensor(); }
  int64_t ndim() const override;
  bool is_cuda() const override;
  int64_t dim(int64_t index) const override;
  int64_t nelement() const override;
  std::shared_ptr<ConsistentTensor> data() const;

  // Getters valid only for EagerMirroredTensor
  Maybe<vm::EagerBlobObject> eager_blob_object() const override {
    return impl_->eager_blob_object();
  }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override {
    return impl_->compute_local_dep_object();
  }
  const TensorMeta& tensor_meta() const override { return *impl_->tensor_meta(); }

  // Setters
  Maybe<void> set_consumer_parallel_distribution_constraint(
      Symbol<cfg::ParallelDistribution> val) override {
    impl_->set_consumer_parallel_distribution_constraint(val);
    return Maybe<void>::Ok();
  }

  // Getters for autograd
  const std::shared_ptr<Tensor>& acc_grad() const override { return impl_->acc_grad(); }
  const std::shared_ptr<TensorArg>& now_grad_arg() const override { return impl_->now_grad_arg(); }
  bool requires_grad() const override { return impl_->requires_grad(); }
  bool is_leaf() const override { return impl_->is_leaf(); }
  bool retain_grad() const override { return impl_->retain_grad(); }

  // Setters for autograd
  void set_acc_grad(const std::shared_ptr<Tensor>& grad) override { impl_->set_acc_grad(grad); }
  std::shared_ptr<Tensor> mut_acc_grad() override { return impl_->mut_acc_grad(); }
  void set_requires_grad(bool requires_grad) override { impl_->set_requires_grad(requires_grad); }
  void set_retain_grad(bool retain_grad) override { impl_->set_retain_grad(retain_grad); }
  void set_is_leaf(bool is_leaf) override { impl_->set_is_leaf(is_leaf); }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() override { return impl_->mut_autograd_meta(); }

  // Operators for tensor
  virtual Maybe<ConsistentTensor> api_detach() const override;

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

 private:
  std::shared_ptr<ConsistentTensorImpl> impl_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
