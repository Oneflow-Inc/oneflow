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
#include "oneflow/core/framework/object.h"
#include "oneflow/core/framework/tensor_storage.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/framework/transport_token.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/eager/local_dep_object.h"

namespace oneflow {

class MemoryCase;

class Shape;
class Stride;
class Device;

namespace vm {
class EagerBlobObject;
class TensorStorage;
}  // namespace vm

namespace one {

class Tensor;
class TensorArg;

class TensorImpl {
 public:
  virtual ~TensorImpl() = default;

  // Getters
  virtual std::shared_ptr<const Shape> shape() const = 0;
  virtual std::shared_ptr<const Stride> stride() const = 0;
  virtual DataType dtype() const = 0;
  virtual bool is_lazy() const = 0;

  // Getters valid only for EagerMirroredTensorImpl
  virtual Maybe<vm::EagerBlobObject> eager_blob_object() const = 0;
  virtual Maybe<LocalDepObject*> compute_local_dep_object() const = 0;
  virtual Maybe<TensorStorage> tensor_storage() const { OF_UNIMPLEMENTED(); }
  virtual Maybe<bool> has_eager_blob_object() const = 0;
  virtual Maybe<int64_t> storage_offset() const { OF_UNIMPLEMENTED(); }
  virtual bool is_contiguous() const = 0;

  // Getters for autograd
  Maybe<Tensor> acc_grad() const;
  Maybe<TensorArg> current_grad() const;
  bool requires_grad() const { return autograd_meta_->requires_grad(); }
  bool is_leaf() const { return autograd_meta_->is_leaf(); }
  bool retain_grad() const { return autograd_meta_->retain_grad(); }

  // Setters for autograd
  Maybe<void> set_acc_grad(const std::shared_ptr<Tensor>& grad);
  Maybe<Tensor> mut_acc_grad();
  Maybe<void> set_requires_grad(bool requires_grad);
  Maybe<void> set_retain_grad(bool retain_grad);

  void set_is_leaf(bool is_leaf) { autograd_meta_->set_is_leaf(is_leaf); }

  std::shared_ptr<const AutogradMeta> autograd_meta() const { return autograd_meta_; }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() { return autograd_meta_; }
  void set_autograd_meta(const std::shared_ptr<AutogradMeta>& autograd_meta) {
    autograd_meta_ = autograd_meta;
  }

  virtual Maybe<void> RegisterStorageDeleteHook(const std::function<void()>& hook) {
    OF_UNIMPLEMENTED();
  }

 protected:
  TensorImpl(bool requires_grad, bool is_leaf)
      : autograd_meta_(std::make_shared<AutogradMeta>(requires_grad, is_leaf)) {}

 protected:
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
  bool is_contiguous() const override { return tensor_meta_->is_contiguous(); }

  // Setters
  MirroredTensorMeta* mut_tensor_meta() {
    return const_cast<MirroredTensorMeta*>(tensor_meta_.get());
  }
  Maybe<Symbol<Device>*> mut_device() { return mut_tensor_meta()->mut_device(); }
  virtual Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() {
    RETURN_ERROR_WITH_BUG_PROMPT();
  }

  virtual Maybe<MirroredTensorImpl> detach() const { RETURN_ERROR_WITH_BUG_PROMPT(); }

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
  std::shared_ptr<const Shape> shape() const override { return tensor_meta_->shape_ptr(); }
  std::shared_ptr<const Stride> stride() const override { return tensor_meta_->stride_ptr(); }
  DataType dtype() const override { return tensor_meta_->dtype(); }
  Symbol<NdSbp> nd_sbp() const { return tensor_meta_->nd_sbp(); }
  Symbol<ParallelDesc> parallel_desc() const { return tensor_meta_->parallel_desc(); }
  const Optional<Symbol<NdSbp>>& consumer_nd_sbp_constraint() const {
    return consumer_nd_sbp_constraint_;
  }
  virtual Maybe<MirroredTensor> cur_rank_phy_tensor() const { RETURN_ERROR_WITH_BUG_PROMPT(); }
  Symbol<ConsistentTensorMeta> tensor_meta() const { return tensor_meta_; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { RETURN_ERROR_WITH_BUG_PROMPT(); }
  Maybe<LocalDepObject*> compute_local_dep_object() const override {
    RETURN_ERROR_WITH_BUG_PROMPT();
  }
  Maybe<bool> has_eager_blob_object() const override { RETURN_ERROR_WITH_BUG_PROMPT(); }

  // Setters
  void set_consumer_nd_sbp_constraint(const Optional<Symbol<NdSbp>>& val) {
    consumer_nd_sbp_constraint_ = val;
  }

  ConsistentTensorMeta* mut_tensor_meta() {
    PRINT_BUG_PROMPT_AND_ABORT();
    return nullptr;
  }

  Maybe<TransportToken> transport_token() const { return JUST(transport_token_); }

  Maybe<void> set_transport_token(const TransportToken& transport_token) {
    transport_token_ = transport_token;
    return Maybe<void>::Ok();
  }

  virtual Maybe<ConsistentTensorImpl> detach() const { RETURN_ERROR_WITH_BUG_PROMPT(); }

 protected:
  ConsistentTensorImpl(Symbol<ConsistentTensorMeta> tensor_meta, bool requires_grad, bool is_leaf)
      : TensorImpl(requires_grad, is_leaf),
        tensor_meta_(tensor_meta),
        consumer_nd_sbp_constraint_(),
        transport_token_() {}

  Symbol<ConsistentTensorMeta> tensor_meta_;
  Optional<Symbol<NdSbp>> consumer_nd_sbp_constraint_;
  Optional<TransportToken> transport_token_;
};

class LazyMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyMirroredTensorImpl);
  LazyMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                         bool requires_grad, bool is_leaf)
      : MirroredTensorImpl(tensor_meta, requires_grad, is_leaf) {}
  ~LazyMirroredTensorImpl() override = default;

  // Getters
  std::shared_ptr<const Shape> shape() const override { return tensor_meta()->shape_ptr(); }
  std::shared_ptr<const Stride> stride() const override { return tensor_meta()->stride_ptr(); }
  bool is_lazy() const override { return true; }
  bool is_contiguous() const override {
    // TODO:(zhaoluyang) default return true for now,
    // but should return real status while stride/view mechanism is ready in lazy-mirrored mode
    return true;
  }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { RETURN_ERROR_WITH_BUG_PROMPT(); }
  Maybe<LocalDepObject*> compute_local_dep_object() const override {
    RETURN_ERROR_WITH_BUG_PROMPT();
  }
  Maybe<TensorStorage> tensor_storage() const override { RETURN_ERROR_WITH_BUG_PROMPT(); }
  Maybe<bool> has_eager_blob_object() const override { RETURN_ERROR_WITH_BUG_PROMPT(); }
  Maybe<MirroredTensorImpl> detach() const override;
};

class EagerMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerMirroredTensorImpl);
  EagerMirroredTensorImpl();
  EagerMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                          bool requires_grad, bool is_leaf);
  EagerMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                          const std::shared_ptr<TensorStorage>& tensor_storage, bool requires_grad,
                          bool is_leaf);
  ~EagerMirroredTensorImpl() override;

  // Getters
  std::shared_ptr<const Shape> shape() const override;
  std::shared_ptr<const Stride> stride() const override;
  Maybe<MirroredTensorImpl> detach() const override;
  bool is_lazy() const override { return false; }
  bool is_contiguous() const override { return tensor_meta_->is_contiguous(); }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override {
    CHECK_OR_RETURN(eager_blob_object_);
    return eager_blob_object_;
  }
  Maybe<LocalDepObject*> compute_local_dep_object() const override;
  Maybe<TensorStorage> tensor_storage() const override {
    CHECK_OR_RETURN(eager_blob_object_);
    return tensor_storage_;
  }
  Maybe<bool> has_eager_blob_object() const override { return eager_blob_object_.get(); }
  Maybe<int64_t> storage_offset() const override { return tensor_meta_->storage_offset(); }

  // Setters
  TensorStorage* mut_tensor_storage() { return tensor_storage_.get(); }

  Maybe<void> InitEagerBlobObject(const intrusive::shared_ptr<LocalDepObject>& dep_object);
  Maybe<void> InitEagerBlobObject(const intrusive::shared_ptr<LocalDepObject>& dep_object,
                                  const bool pin_memory);
  Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() override { return this; }

  Maybe<void> RegisterStorageDeleteHook(const std::function<void()>& hook) override;

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

  bool is_contiguous() const override {
    // TODO:(zhaoluyang) default return true for now,
    // but should return real status while stride/view mechanism is ready in lazy-consistent mode
    return true;
  }

  Maybe<ConsistentTensorImpl> detach() const override;
};

class EagerConsistentTensorImpl final : public ConsistentTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerConsistentTensorImpl);
  ~EagerConsistentTensorImpl() override = default;

  // Getters
  std::shared_ptr<const Stride> stride() const override;
  bool is_lazy() const override { return false; }

  bool is_contiguous() const override {
    // TODO:(zhaoluyang) default return true for now,
    // but should return real status while stride/view mechanism is ready in eager-consistent mode
    return true;
  }

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

  Maybe<ConsistentTensorImpl> detach() const override;

 private:
  EagerConsistentTensorImpl(Symbol<ConsistentTensorMeta> consistent_tensor_meta, bool requires_grad,
                            bool is_leaf,
                            const std::shared_ptr<MirroredTensor>& cur_rank_phy_tensor);

  std::shared_ptr<MirroredTensor> cur_rank_phy_tensor_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_
