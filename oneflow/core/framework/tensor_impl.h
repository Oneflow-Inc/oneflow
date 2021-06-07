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

#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/framework/object.h"
#include "oneflow/core/framework/tensor_storage.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class MemoryCase;
class VmLocalDepObject;

namespace cfg {

class ParallelDistribution;
}

class Shape;
class Device;

namespace vm {
class EagerBlobObject;
}  // namespace vm

namespace one {

class Tensor;
class TensorArg;

class TensorImpl {
 public:
  virtual ~TensorImpl() = default;

  // Getters
  virtual const std::shared_ptr<const Shape>& shape() const = 0;
  virtual DataType dtype() const = 0;
  virtual bool is_lazy() const = 0;

  // Getters valid only for EagerMirroredTensorImpl
  virtual Maybe<vm::EagerBlobObject> eager_blob_object() const = 0;
  virtual Maybe<VmLocalDepObject> compute_local_dep_object() const = 0;

  // Getters for autograd
  const std::shared_ptr<Tensor>& acc_grad() const { return autograd_meta_->acc_grad(); }
  const std::shared_ptr<TensorArg>& now_grad_arg() const { return autograd_meta_->now_grad_arg(); }
  bool requires_grad() const { return autograd_meta_->requires_grad(); }
  bool is_leaf() const { return autograd_meta_->is_leaf(); }
  bool retain_grad() const { return autograd_meta_->retain_grad(); }

  // Setters for autograd
  void set_acc_grad(const std::shared_ptr<Tensor>& grad) { autograd_meta_->set_acc_grad(grad); }
  std::shared_ptr<Tensor> mut_acc_grad() { return autograd_meta_->mut_acc_grad(); }
  void set_requires_grad(bool requires_grad) { autograd_meta_->set_requires_grad(requires_grad); }
  void set_retain_grad(bool retain_grad) { autograd_meta_->set_retain_grad(retain_grad); }
  void set_is_leaf(bool is_leaf) { autograd_meta_->set_is_leaf(is_leaf); }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() { return autograd_meta_; }

 protected:
  TensorImpl(const std::shared_ptr<AutogradMeta>& autograd_meta) : autograd_meta_(autograd_meta) {}

 protected:
  std::shared_ptr<AutogradMeta> autograd_meta_;
};

class EagerMirroredTensorImpl;

class MirroredTensorImpl : public TensorImpl {
 public:
  virtual ~MirroredTensorImpl() = default;

  // Getters
  DataType dtype() const override { return tensor_meta_->dtype(); }
  const std::shared_ptr<const Device>& device() const { return tensor_meta_->device(); }
  const std::shared_ptr<const MirroredTensorMeta>& tensor_meta() const { return tensor_meta_; }

  // Setters
  MirroredTensorMeta* mut_tensor_meta() {
    return const_cast<MirroredTensorMeta*>(tensor_meta_.get());
  }
  Maybe<std::shared_ptr<const Device>*> mut_device() { return mut_tensor_meta()->mut_device(); }
  Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() { OF_UNIMPLEMENTED(); }

 protected:
  MirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                     const std::shared_ptr<AutogradMeta>& autograd_meta)
      : TensorImpl(autograd_meta), tensor_meta_(tensor_meta) {}

  std::shared_ptr<const MirroredTensorMeta> tensor_meta_;
};

class MirroredTensor;

class ConsistentTensorImpl : public TensorImpl {
 public:
  virtual ~ConsistentTensorImpl() = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return tensor_meta_->shape_ptr(); }
  DataType dtype() const override { return tensor_meta_->dtype(); }
  Symbol<cfg::ParallelDistribution> parallel_distribution() const {
    return tensor_meta_->parallel_distribution();
  }
  Symbol<ParallelDesc> parallel_desc() const { return tensor_meta_->parallel_desc(); }
  Symbol<cfg::ParallelDistribution> consumer_parallel_distribution_constraint() const {
    return consumer_parallel_distribution_constraint_;
  }
  virtual Maybe<MirroredTensor> cur_rank_phy_tensor() const { OF_UNIMPLEMENTED(); }
  Symbol<ConsistentTensorMeta> tensor_meta() const { return tensor_meta_; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override { OF_UNIMPLEMENTED(); }

  // Setters
  void set_consumer_parallel_distribution_constraint(Symbol<cfg::ParallelDistribution> val) {
    consumer_parallel_distribution_constraint_ = val;
  }

  ConsistentTensorMeta* mut_tensor_meta() {
    UNIMPLEMENTED();
    return nullptr;
  }

 protected:
  ConsistentTensorImpl(Symbol<ConsistentTensorMeta> tensor_meta,
                       const std::shared_ptr<AutogradMeta>& autograd_meta)
      : TensorImpl(autograd_meta),
        tensor_meta_(tensor_meta),
        consumer_parallel_distribution_constraint_() {}

  Symbol<ConsistentTensorMeta> tensor_meta_;
  Symbol<cfg::ParallelDistribution> consumer_parallel_distribution_constraint_;
};

class LazyMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyMirroredTensorImpl);
  LazyMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                         bool requires_grad, bool is_leaf)
      : MirroredTensorImpl(tensor_meta, NewAutogradMeta(requires_grad, is_leaf)) {}
  ~LazyMirroredTensorImpl() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const { return tensor_meta()->shape_ptr(); }
  bool is_lazy() const override { return true; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override { OF_UNIMPLEMENTED(); }
};

class EagerMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerMirroredTensorImpl);
  EagerMirroredTensorImpl();
  EagerMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                          const std::shared_ptr<AutogradMeta>& autograd_meta);
  EagerMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                          bool requires_grad, bool is_leaf);
  ~EagerMirroredTensorImpl() override;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override;
  bool is_lazy() const override { return false; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { return eager_blob_object_; }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override;

  // Setters
  TensorStorage* mut_tensor_storage() { return tensor_storage_.get(); }

  Maybe<void> InitEagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case);
  Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() { return this; }

 private:
  void UpdateTensorStorage();
  Maybe<void> set_eager_blob_object(std::shared_ptr<vm::EagerBlobObject> eager_blob_object);

  std::shared_ptr<TensorStorage> tensor_storage_;
  std::shared_ptr<vm::EagerBlobObject> eager_blob_object_;
};

class LazyConsistentTensorImpl final : public ConsistentTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyConsistentTensorImpl);
  LazyConsistentTensorImpl(Symbol<ConsistentTensorMeta> consistent_tensor_meta, bool requires_grad,
                           bool is_leaf)
      : ConsistentTensorImpl(consistent_tensor_meta, NewAutogradMeta(requires_grad, is_leaf)) {}
  ~LazyConsistentTensorImpl() override = default;

  // Getters
  bool is_lazy() const override { return true; }
};

class EagerConsistentTensorImpl final : public ConsistentTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerConsistentTensorImpl);
  ~EagerConsistentTensorImpl() override = default;

  // Getters
  bool is_lazy() const override { return false; }

  Maybe<MirroredTensor> cur_rank_phy_tensor() const override { return cur_rank_phy_tensor_; }

  static Maybe<EagerConsistentTensorImpl> New(
      const std::shared_ptr<MirroredTensor>& cur_rank_phy_tensor,
      Symbol<cfg::ParallelDistribution> parallel_distribution, Symbol<ParallelDesc> parallel_desc);

  static Maybe<EagerConsistentTensorImpl> New(Symbol<ConsistentTensorMeta> consistent_tensor_meta,
                                              bool requires_grad, bool is_leaf);

  static Maybe<EagerConsistentTensorImpl> NewWithPhyTensor(
      Symbol<ConsistentTensorMeta> consistent_tensor_meta,
      const std::shared_ptr<const Device>& device, int64_t parallel_id, bool requires_grad,
      bool is_leaf);

  static Maybe<EagerConsistentTensorImpl> NewWithoutPhyTensor(
      Symbol<ConsistentTensorMeta> consistent_tensor_meta,
      const std::shared_ptr<const Device>& device, int64_t parallel_id, bool requires_grad,
      bool is_leaf);

  typedef Maybe<EagerConsistentTensorImpl> (*NewMethod)(Symbol<ConsistentTensorMeta>,
                                                        const std::shared_ptr<const Device>&,
                                                        int64_t, bool, bool);

 private:
  EagerConsistentTensorImpl(Symbol<ConsistentTensorMeta> consistent_tensor_meta,
                            const std::shared_ptr<AutogradMeta>& autograd_meta,
                            const std::shared_ptr<MirroredTensor>& cur_rank_phy_tensor);

  std::shared_ptr<MirroredTensor> cur_rank_phy_tensor_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_
