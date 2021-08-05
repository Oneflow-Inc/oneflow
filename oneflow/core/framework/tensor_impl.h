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
#include "oneflow/core/common/optional.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/framework/object.h"
#include "oneflow/core/framework/tensor_storage.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/framework/transport_token.h"
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
  virtual Maybe<TensorStorage> tensor_storage() const { OF_UNIMPLEMENTED(); }
  virtual Maybe<bool> has_eager_blob_object() const = 0;
  virtual Maybe<const Stride> stride() const { OF_UNIMPLEMENTED(); }
  virtual Maybe<int64_t> storage_offset() const { OF_UNIMPLEMENTED(); }

  // Getters for autograd
  Maybe<Tensor> acc_grad() const;
  Maybe<TensorArg> current_grad() const;
  bool requires_grad() const { return requires_grad_; }
  bool is_leaf() const { return is_leaf_; }
  bool retain_grad() const { return autograd_meta_->retain_grad(); }

  // Setters for autograd
  Maybe<void> set_acc_grad(const std::shared_ptr<Tensor>& grad);
  Maybe<Tensor> mut_acc_grad();
  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
  Maybe<void> set_retain_grad(bool retain_grad);
  void set_is_leaf(bool is_leaf) { is_leaf_ = is_leaf; }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() { return autograd_meta_; }
  void set_autograd_meta(const std::shared_ptr<AutogradMeta>& autograd_meta) {
    autograd_meta_ = autograd_meta;
  }
  bool has_autograd_meta() const { return autograd_meta_.get(); }

 protected:
  TensorImpl(bool requires_grad, bool is_leaf) : requires_grad_(requires_grad), is_leaf_(is_leaf) {}

 protected:
  bool requires_grad_;
  bool is_leaf_;
  std::shared_ptr<AutogradMeta> autograd_meta_;
};

class EagerMirroredTensorImpl;
class MirroredTensorImpl : public TensorImpl {
 public:
  virtual ~MirroredTensorImpl() = default;

  // Getters
  DataType dtype() const override { return tensor_meta_->dtype(); }
  const Symbol<Device>& device() const { return tensor_meta_->device(); }
  const std::shared_ptr<const MirroredTensorMeta>& tensor_meta() const { return tensor_meta_; }

  // Setters
  MirroredTensorMeta* mut_tensor_meta() {
    return const_cast<MirroredTensorMeta*>(tensor_meta_.get());
  }
  Maybe<Symbol<Device>*> mut_device() { return mut_tensor_meta()->mut_device(); }
  virtual Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() { OF_UNIMPLEMENTED(); }

  virtual Maybe<MirroredTensorImpl> detach() const { OF_UNIMPLEMENTED(); }

 protected:
  MirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                     bool requires_grad, bool is_leaf)
      : TensorImpl(requires_grad, is_leaf), tensor_meta_(tensor_meta) {}

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
  const Optional<Symbol<cfg::ParallelDistribution>>& consumer_parallel_distribution_constraint()
      const {
    return consumer_parallel_distribution_constraint_;
  }
  virtual Maybe<MirroredTensor> cur_rank_phy_tensor() const { OF_UNIMPLEMENTED(); }
  Symbol<ConsistentTensorMeta> tensor_meta() const { return tensor_meta_; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<bool> has_eager_blob_object() const override { OF_UNIMPLEMENTED(); }

  // Setters
  void set_consumer_parallel_distribution_constraint(Symbol<cfg::ParallelDistribution> val) {
    consumer_parallel_distribution_constraint_ = val;
  }

  ConsistentTensorMeta* mut_tensor_meta() {
    UNIMPLEMENTED();
    return nullptr;
  }

  const Maybe<TransportToken> transport_token() const { return transport_token_; }

  Maybe<void> set_transport_token(const TransportToken& transport_token) {
    CHECK_OR_RETURN(!transport_token_.IsOk()) << "transport_token_ is initiliazed";
    transport_token_ = transport_token;
    return Maybe<void>::Ok();
  }

 protected:
  ConsistentTensorImpl(Symbol<ConsistentTensorMeta> tensor_meta, bool requires_grad, bool is_leaf)
      : TensorImpl(requires_grad, is_leaf),
        tensor_meta_(tensor_meta),
        consumer_parallel_distribution_constraint_(),
        transport_token_(Error::ValueError("invalid rpc token")) {}

  Symbol<ConsistentTensorMeta> tensor_meta_;
  Optional<Symbol<cfg::ParallelDistribution>> consumer_parallel_distribution_constraint_;
  Maybe<TransportToken> transport_token_;
};

class LazyMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyMirroredTensorImpl);
  LazyMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                         bool requires_grad, bool is_leaf)
      : MirroredTensorImpl(tensor_meta, requires_grad, is_leaf) {}
  ~LazyMirroredTensorImpl() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return tensor_meta()->shape_ptr(); }
  bool is_lazy() const override { return true; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<TensorStorage> tensor_storage() const override { OF_UNIMPLEMENTED(); }
  Maybe<bool> has_eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<MirroredTensorImpl> detach() const override;
};

class EagerMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerMirroredTensorImpl);
  EagerMirroredTensorImpl();
  EagerMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                          bool requires_grad, bool is_leaf);
  EagerMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                          const std::shared_ptr<TensorStorage> tensor_storage, bool requires_grad,
                          bool is_leaf);
  ~EagerMirroredTensorImpl() override;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override;
  Maybe<MirroredTensorImpl> detach() const override;
  bool is_lazy() const override { return false; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override {
    CHECK_OR_RETURN(eager_blob_object_);
    return eager_blob_object_;
  }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override;
  Maybe<TensorStorage> tensor_storage() const override {
    CHECK_OR_RETURN(eager_blob_object_);
    return tensor_storage_;
  }
  Maybe<bool> has_eager_blob_object() const override { return eager_blob_object_.get(); }
  Maybe<const Stride> stride() const override { return tensor_meta_->stride_ptr(); }
  Maybe<int64_t> storage_offset() const override { return tensor_meta_->storage_offset(); }

  // Setters
  TensorStorage* mut_tensor_storage() { return tensor_storage_.get(); }

  Maybe<void> InitEagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case);
  Maybe<void> InitEagerBlobObjectAndTensorStorage(
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
      const std::shared_ptr<TensorStorage>& tensor_storage);
  Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() override { return this; }

 private:
  Maybe<void> UpdateTensorStorage();
  Maybe<void> set_eager_blob_object(std::shared_ptr<vm::EagerBlobObject> eager_blob_object);

  std::shared_ptr<TensorStorage> tensor_storage_;
  std::shared_ptr<vm::EagerBlobObject> eager_blob_object_;
};

class LazyConsistentTensorImpl final : public ConsistentTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyConsistentTensorImpl);
  LazyConsistentTensorImpl(Symbol<ConsistentTensorMeta> consistent_tensor_meta, bool requires_grad,
                           bool is_leaf)
      : ConsistentTensorImpl(consistent_tensor_meta, requires_grad, is_leaf) {}
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
  void reset_cur_rank_phy_tensor(const std::shared_ptr<MirroredTensor>& val) {
    cur_rank_phy_tensor_ = val;
  }

  static Maybe<EagerConsistentTensorImpl> New(Symbol<ConsistentTensorMeta> consistent_tensor_meta,
                                              bool requires_grad, bool is_leaf);

  static Maybe<EagerConsistentTensorImpl> New(Symbol<ConsistentTensorMeta> consistent_tensor_meta,
                                              Symbol<Device> device,
                                              const Optional<int64_t>& parallel_id,
                                              bool requires_grad, bool is_leaf);

 private:
  EagerConsistentTensorImpl(Symbol<ConsistentTensorMeta> consistent_tensor_meta, bool requires_grad,
                            bool is_leaf,
                            const std::shared_ptr<MirroredTensor>& cur_rank_phy_tensor);

  std::shared_ptr<MirroredTensor> cur_rank_phy_tensor_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_
