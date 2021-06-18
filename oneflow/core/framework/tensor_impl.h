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

class TensorMeta : public user_op::TensorDesc {
 public:
  TensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype)
      : shape_(shape), data_type_(dtype), is_dynamic_(false) {}
  TensorMeta(const TensorMeta&) = default;
  TensorMeta(TensorMeta&&) = default;
  ~TensorMeta() = default;

  const std::shared_ptr<const Shape>& shape_ptr() const { return shape_; }

  const Shape& shape() const override { return *shape_; }
  DataType dtype() const { return data_type_; }
  DataType data_type() const override { return data_type_; }
  bool is_dynamic() const override { return is_dynamic_; }

  void set_shape(const std::shared_ptr<const Shape>& val) { shape_ = val; }
  Shape* mut_shape() override { return const_cast<Shape*>(shape_.get()); }
  DataType* mut_dtype() { return &data_type_; }
  void set_dtype(DataType data_type) { data_type_ = data_type; }
  DataType* mut_data_type() override { return &data_type_; }
  bool* mut_is_dynamic() override { return &is_dynamic_; }
  void set_is_dynamic(bool val) override { is_dynamic_ = val; }

 private:
  std::shared_ptr<const Shape> shape_;
  DataType data_type_;
  bool is_dynamic_;
};

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
  virtual Maybe<TensorStorage> tensor_storage() const = 0;

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

class MirroredTensorMeta : public TensorMeta {
 public:
  MirroredTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                     const std::shared_ptr<const Device>& device)
      : TensorMeta(shape, dtype), device_(device) {}

  const std::shared_ptr<const Device>& device() const { return device_; }

  std::shared_ptr<const Device>* mut_device() { return &device_; }

  bool operator==(const MirroredTensorMeta& other) const;
  size_t CalcHashValue() const;

 private:
  std::shared_ptr<const Device> device_;
};

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
  std::shared_ptr<const Device>* mut_device() { return mut_tensor_meta()->mut_device(); }
  virtual Maybe<void> set_tensor_storage(std::shared_ptr<TensorStorage> tensor_storage) = 0;

  virtual Maybe<MirroredTensorImpl> detach() const { OF_UNIMPLEMENTED(); }

 protected:
  MirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                     const std::shared_ptr<AutogradMeta>& autograd_meta)
      : TensorImpl(autograd_meta), tensor_meta_(tensor_meta) {}

  std::shared_ptr<const MirroredTensorMeta> tensor_meta_;
};

class ConsistentTensorMeta : public TensorMeta {
 public:
  ConsistentTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                       Symbol<cfg::ParallelDistribution> parallel_distribution,
                       Symbol<ParallelDesc> parallel_desc)
      : TensorMeta(shape, dtype),
        parallel_distribution_(parallel_distribution),
        parallel_desc_(parallel_desc) {}
  ConsistentTensorMeta(const ConsistentTensorMeta&) = default;
  ConsistentTensorMeta(ConsistentTensorMeta&&) = default;
  ~ConsistentTensorMeta() = default;

  bool operator==(const ConsistentTensorMeta& other) const;

  Symbol<cfg::ParallelDistribution> parallel_distribution() const { return parallel_distribution_; }
  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }

  size_t CalcHashValue() const;

 private:
  Symbol<cfg::ParallelDistribution> parallel_distribution_;
  Symbol<ParallelDesc> parallel_desc_;
};

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
  Symbol<cfg::ParallelDistribution> consumer_forced_parallel_distribution() const {
    return consumer_forced_parallel_distribution_;
  }
  Symbol<ConsistentTensorMeta> tensor_meta() const { return tensor_meta_; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<TensorStorage> tensor_storage() const override { OF_UNIMPLEMENTED(); }

  // Setters
  void set_consumer_forced_parallel_distribution(Symbol<cfg::ParallelDistribution> val) {
    consumer_forced_parallel_distribution_ = val;
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
        consumer_forced_parallel_distribution_() {}

  Symbol<ConsistentTensorMeta> tensor_meta_;
  Symbol<cfg::ParallelDistribution> consumer_forced_parallel_distribution_;
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
  Maybe<TensorStorage> tensor_storage() const override { OF_UNIMPLEMENTED(); }

  Maybe<void> set_tensor_storage(std::shared_ptr<TensorStorage> tensor_storage) override {
    return Error::Unimplemented();
  }
};

class EagerMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerMirroredTensorImpl);
  EagerMirroredTensorImpl();
  EagerMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                          const std::shared_ptr<AutogradMeta>& autograd_meta);
  EagerMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                          bool requires_grad, bool is_leaf);
  EagerMirroredTensorImpl(const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
                          const std::shared_ptr<TensorStorage> tensor_storage, bool requires_grad,
                          bool is_leaf);
  ~EagerMirroredTensorImpl() override;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override;
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

  // Setters
  Maybe<void> set_tensor_storage(std::shared_ptr<TensorStorage> tensor_storage) override {
    CHECK_OR_RETURN(!tensor_storage_);
    tensor_storage_ = tensor_storage;
    return Maybe<void>::Ok();
  }

  Maybe<void> InitEagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case);

  Maybe<MirroredTensorImpl> detach() const override {
    auto* detached_impl = new EagerMirroredTensorImpl(tensor_meta_, tensor_storage_, false, true);
    detached_impl->eager_blob_object_ = eager_blob_object_;
    return std::shared_ptr<MirroredTensorImpl>(detached_impl);
  }

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
      : ConsistentTensorImpl(consistent_tensor_meta, NewAutogradMeta(requires_grad, is_leaf)) {}
  ~LazyConsistentTensorImpl() override = default;

  // Getters
  bool is_lazy() const override { return true; }
};

class MirroredTensor;

class EagerConsistentTensorImpl final : public ConsistentTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerConsistentTensorImpl);
  ~EagerConsistentTensorImpl() override = default;

  // Getters
  bool is_lazy() const override { return false; }

  const std::shared_ptr<MirroredTensor>& cur_rank_phy_tensor() const {
    return cur_rank_phy_tensor_;
  }

  static Maybe<EagerConsistentTensorImpl> New(
      const std::shared_ptr<MirroredTensor>& cur_rank_phy_tensor,
      Symbol<cfg::ParallelDistribution> parallel_distribution, Symbol<ParallelDesc> parallel_desc);

  static Maybe<EagerConsistentTensorImpl> New(Symbol<ConsistentTensorMeta> consistent_tensor_meta,
                                              bool requires_grad, bool is_leaf);

 private:
  EagerConsistentTensorImpl(Symbol<ConsistentTensorMeta> consistent_tensor_meta,
                            const std::shared_ptr<MirroredTensor>& cur_rank_phy_tensor);

  std::shared_ptr<MirroredTensor> cur_rank_phy_tensor_;
};

}  // namespace one

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::one::ConsistentTensorMeta> final {
  size_t operator()(const oneflow::one::ConsistentTensorMeta& other) const {
    return other.CalcHashValue();
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_
