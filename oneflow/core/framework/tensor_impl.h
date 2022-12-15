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
#include "oneflow/core/framework/tensor_storage.h"
#include "oneflow/core/common/tensor_desc.h"
#include "oneflow/core/common/tensor_meta.h"
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

  // Getters valid only for EagerLocalTensorImpl
  virtual Maybe<vm::EagerBlobObject> eager_blob_object() const = 0;
  virtual Maybe<LocalDepObject*> compute_local_dep_object() const = 0;
  virtual Maybe<TensorStorage> tensor_storage() const { OF_UNIMPLEMENTED(); }
  virtual Maybe<bool> has_eager_blob_object() const = 0;
  virtual Maybe<int64_t> storage_offset() const { OF_UNIMPLEMENTED(); }
  virtual bool is_contiguous() const = 0;
  virtual Maybe<bool> is_pinned() const { OF_UNIMPLEMENTED(); }

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

class EagerLocalTensorImpl;
class LocalTensorImpl : public TensorImpl {
 public:
  virtual ~LocalTensorImpl() = default;

  // Getters
  DataType dtype() const override { return tensor_meta()->dtype(); }
  const Symbol<Device>& device() const { return tensor_meta()->device(); }
  bool is_contiguous() const override { return tensor_meta()->is_contiguous(); }

  virtual const Symbol<LocalTensorMeta>& tensor_meta() const = 0;
  // Setters
  virtual const std::shared_ptr<const MutLocalTensorMeta>& mut_tensor_meta() = 0;
  Maybe<Symbol<Device>*> mut_device() {
    return std::const_pointer_cast<MutLocalTensorMeta>(mut_tensor_meta())->mut_device();
  }
  virtual Maybe<EagerLocalTensorImpl*> mut_eager_local_tensor_impl() {
    RETURN_ERROR_WITH_BUG_PROMPT();
  }

  virtual Maybe<LocalTensorImpl> detach() const { RETURN_ERROR_WITH_BUG_PROMPT(); }

 protected:
  LocalTensorImpl(bool requires_grad, bool is_leaf) : TensorImpl(requires_grad, is_leaf) {}
};

class LocalTensor;

class GlobalTensorImpl : public TensorImpl {
 public:
  virtual ~GlobalTensorImpl() = default;

  // Getters
  std::shared_ptr<const Shape> shape() const override { return tensor_meta_->shape_ptr(); }
  std::shared_ptr<const Stride> stride() const override { return tensor_meta_->stride_ptr(); }
  DataType dtype() const override { return tensor_meta_->dtype(); }
  Symbol<NdSbp> nd_sbp() const { return tensor_meta_->nd_sbp(); }
  Symbol<ParallelDesc> parallel_desc() const { return tensor_meta_->parallel_desc(); }
  const Optional<Symbol<NdSbp>>& consumer_nd_sbp_constraint() const {
    return consumer_nd_sbp_constraint_;
  }
  virtual Maybe<LocalTensor> cur_rank_phy_tensor() const { RETURN_ERROR_WITH_BUG_PROMPT(); }
  Symbol<GlobalTensorMeta> tensor_meta() const { return tensor_meta_; }

  // Getters valid only for EagerLocalTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { RETURN_ERROR_WITH_BUG_PROMPT(); }
  Maybe<LocalDepObject*> compute_local_dep_object() const override {
    RETURN_ERROR_WITH_BUG_PROMPT();
  }
  Maybe<bool> has_eager_blob_object() const override { RETURN_ERROR_WITH_BUG_PROMPT(); }

  // Setters
  void set_consumer_nd_sbp_constraint(const Optional<Symbol<NdSbp>>& val) {
    consumer_nd_sbp_constraint_ = val;
  }

  GlobalTensorMeta* mut_tensor_meta() {
    PRINT_BUG_PROMPT_AND_ABORT();
    return nullptr;
  }

  Maybe<TransportToken> transport_token() const { return JUST(transport_token_); }

  Maybe<void> set_transport_token(const TransportToken& transport_token) {
    transport_token_ = transport_token;
    return Maybe<void>::Ok();
  }

  virtual Maybe<GlobalTensorImpl> detach() const { RETURN_ERROR_WITH_BUG_PROMPT(); }

 protected:
  GlobalTensorImpl(Symbol<GlobalTensorMeta> tensor_meta, bool requires_grad, bool is_leaf)
      : TensorImpl(requires_grad, is_leaf),
        tensor_meta_(tensor_meta),
        consumer_nd_sbp_constraint_(),
        transport_token_() {}

  Symbol<GlobalTensorMeta> tensor_meta_;
  Optional<Symbol<NdSbp>> consumer_nd_sbp_constraint_;
  Optional<TransportToken> transport_token_;
};

class LazyLocalTensorImpl final : public LocalTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyLocalTensorImpl);
  LazyLocalTensorImpl(const Symbol<LocalTensorMeta>& tensor_meta, bool requires_grad, bool is_leaf)
      : LocalTensorImpl(requires_grad, is_leaf), tensor_meta_(tensor_meta) {}
  ~LazyLocalTensorImpl() override = default;

  // Getters
  const Symbol<LocalTensorMeta>& tensor_meta() const override { return tensor_meta_; }
  std::shared_ptr<const Shape> shape() const override { return tensor_meta()->shape_ptr(); }
  std::shared_ptr<const Stride> stride() const override { return tensor_meta()->stride_ptr(); }
  bool is_lazy() const override { return true; }
  bool is_contiguous() const override {
    // TODO:(zhaoluyang) default return true for now,
    // but should return real status while stride/view mechanism is ready in lazy-local mode
    return true;
  }
  Maybe<bool> is_pinned() const override { return false; }

  const std::shared_ptr<const MutLocalTensorMeta>& mut_tensor_meta() override {
    PRINT_BUG_PROMPT_AND_ABORT();
  }

  // Getters valid only for EagerLocalTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { RETURN_ERROR_WITH_BUG_PROMPT(); }
  Maybe<LocalDepObject*> compute_local_dep_object() const override {
    RETURN_ERROR_WITH_BUG_PROMPT();
  }
  Maybe<TensorStorage> tensor_storage() const override { RETURN_ERROR_WITH_BUG_PROMPT(); }
  Maybe<bool> has_eager_blob_object() const override { RETURN_ERROR_WITH_BUG_PROMPT(); }
  Maybe<LocalTensorImpl> detach() const override;

 private:
  Symbol<LocalTensorMeta> tensor_meta_;
};

class EagerLocalTensorImpl final : public LocalTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerLocalTensorImpl);
  EagerLocalTensorImpl()
      : EagerLocalTensorImpl(std::shared_ptr<TensorStorage>(), 0, false, false) {}
  EagerLocalTensorImpl(const std::shared_ptr<TensorStorage>& tensor_storage, bool requires_grad,
                       bool is_leaf)
      : EagerLocalTensorImpl(tensor_storage, 0, requires_grad, is_leaf) {}
  EagerLocalTensorImpl(const std::shared_ptr<TensorStorage>& tensor_storage, int64_t storage_offset,
                       bool requires_grad, bool is_leaf);

  EagerLocalTensorImpl(bool requires_grad, bool is_leaf)
      : EagerLocalTensorImpl(std::shared_ptr<TensorStorage>(), 0, requires_grad, is_leaf) {}
  ~EagerLocalTensorImpl() override;

  const std::shared_ptr<const MutLocalTensorMeta>& mut_tensor_meta() override;
  // Getters
  const Symbol<LocalTensorMeta>& tensor_meta() const override;
  std::shared_ptr<const Shape> shape() const override;
  std::shared_ptr<const Stride> stride() const override;
  Maybe<LocalTensorImpl> detach() const override;
  bool is_lazy() const override { return false; }
  bool is_contiguous() const override { return tensor_meta()->is_contiguous(); }
  Maybe<bool> is_pinned() const override;

  // Getters valid only for EagerLocalTensorImpl
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
  Maybe<int64_t> storage_offset() const override { return storage_offset_; }
  // Setters
  TensorStorage* mut_tensor_storage() { return tensor_storage_.get(); }
  void set_storage_offset(int64_t offset) { storage_offset_ = offset; }

  Maybe<void> InitEagerBlobObject(
      const Symbol<one::LocalTensorMeta>& local_tensor_meta,
      const std::shared_ptr<const one::MutLocalTensorMeta>& mut_local_tensor_meta,
      const intrusive::shared_ptr<LocalDepObject>& dep_object);
  Maybe<void> InitEagerBlobObject(const Symbol<one::LocalTensorMeta>& local_tensor_meta,
                                  const intrusive::shared_ptr<LocalDepObject>& dep_object) {
    JUST(InitEagerBlobObject(local_tensor_meta, std::shared_ptr<const one::MutLocalTensorMeta>(),
                             dep_object));
    return Maybe<void>::Ok();
  }

  Maybe<EagerLocalTensorImpl*> mut_eager_local_tensor_impl() override { return this; }

  Maybe<void> RegisterStorageDeleteHook(const std::function<void()>& hook) override;

 private:
  Maybe<void> UpdateTensorStorage();
  Maybe<void> set_eager_blob_object(std::shared_ptr<vm::EagerBlobObject> eager_blob_object);

  std::shared_ptr<TensorStorage> tensor_storage_;
  int64_t storage_offset_;
  std::shared_ptr<vm::EagerBlobObject> eager_blob_object_;
};

class LazyGlobalTensorImpl final : public GlobalTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyGlobalTensorImpl);
  LazyGlobalTensorImpl(Symbol<GlobalTensorMeta> global_tensor_meta, bool requires_grad,
                       bool is_leaf)
      : GlobalTensorImpl(global_tensor_meta, requires_grad, is_leaf) {}
  ~LazyGlobalTensorImpl() override = default;

  // Getters
  bool is_lazy() const override { return true; }

  bool is_contiguous() const override {
    // TODO:(zhaoluyang) default return true for now,
    // but should return real status while stride/view mechanism is ready in lazy-global mode
    return true;
  }

  Maybe<GlobalTensorImpl> detach() const override;
};

class EagerGlobalTensorImpl final : public GlobalTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerGlobalTensorImpl);
  ~EagerGlobalTensorImpl() override = default;

  // Getters
  std::shared_ptr<const Stride> stride() const override;
  bool is_lazy() const override { return false; }

  bool is_contiguous() const override {
    // TODO:(zhaoluyang) default return true for now,
    // but should return real status while stride/view mechanism is ready in eager-global mode
    return true;
  }

  Maybe<LocalTensor> cur_rank_phy_tensor() const override { return cur_rank_phy_tensor_; }
  void reset_cur_rank_phy_tensor(const std::shared_ptr<LocalTensor>& val) {
    cur_rank_phy_tensor_ = val;
  }

  static Maybe<EagerGlobalTensorImpl> New(Symbol<GlobalTensorMeta> global_tensor_meta,
                                          bool requires_grad, bool is_leaf);

  static Maybe<EagerGlobalTensorImpl> New(Symbol<GlobalTensorMeta> global_tensor_meta,
                                          Symbol<Device> device,
                                          const Optional<int64_t>& parallel_id, bool requires_grad,
                                          bool is_leaf);

  Maybe<GlobalTensorImpl> detach() const override;

 private:
  EagerGlobalTensorImpl(Symbol<GlobalTensorMeta> global_tensor_meta,
                        const std::shared_ptr<LocalTensor>& cur_rank_phy_tensor);

  std::shared_ptr<LocalTensor> cur_rank_phy_tensor_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_
