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
  virtual Maybe<TensorStorage> tensor_storage() const = 0;

  // Getters for autograd
  Maybe<Tensor> acc_grad() const;
  Maybe<TensorArg> now_grad_arg() const;
  bool requires_grad() const { return requires_grad_; }
  bool is_leaf() const { return is_leaf_; }
  bool retain_grad() const { return autograd_meta_->retain_grad(); }

  // Setters
  virtual void set_shape(const std::shared_ptr<const Shape>& shape) = 0;
  virtual void set_dtype(DataType dtype) = 0;

  // Setters for autograd
  Maybe<void> set_acc_grad(const std::shared_ptr<Tensor>& grad);
  Maybe<Tensor> mut_acc_grad();
  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
  Maybe<void> set_retain_grad(bool retain_grad);
  void set_is_leaf(bool is_leaf) { is_leaf_ = is_leaf; }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() { return autograd_meta_; }

 protected:
  TensorImpl(bool requires_grad, bool is_leaf) : requires_grad_(requires_grad), is_leaf_(is_leaf) {}

 protected:
  bool requires_grad_;
  bool is_leaf_;
  std::shared_ptr<AutogradMeta> autograd_meta_;
};

class MirroredTensorImpl : public TensorImpl {
 public:
  virtual ~MirroredTensorImpl() = default;

  // Getters
  const std::shared_ptr<const Device>& device() const { return device_; }

  // Setters
  Maybe<void> set_device(const std::shared_ptr<const Device>& device);
  virtual Maybe<void> set_eager_blob_object(
      std::shared_ptr<vm::EagerBlobObject> eager_blob_object) = 0;
  virtual Maybe<void> set_tensor_storage(std::shared_ptr<TensorStorage> tensor_storage) = 0;

 protected:
  MirroredTensorImpl(const std::shared_ptr<const Device>& device, bool requires_grad, bool is_leaf)
      : TensorImpl(requires_grad, is_leaf) {
    set_device(device);
  }

  std::shared_ptr<const Device> device_;
};

class ConsistentTensorMeta {
 public:
  ConsistentTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                       Symbol<cfg::ParallelDistribution> parallel_distribution,
                       Symbol<ParallelDesc> parallel_desc)
      : shape_(shape),
        dtype_(dtype),
        parallel_distribution_(parallel_distribution),
        parallel_desc_(parallel_desc) {}
  ConsistentTensorMeta(const ConsistentTensorMeta&) = default;
  ConsistentTensorMeta(ConsistentTensorMeta&&) = default;
  ~ConsistentTensorMeta() = default;

  bool operator==(const ConsistentTensorMeta& other) const;

  const std::shared_ptr<const Shape>& shape() const { return shape_; }
  DataType dtype() const { return dtype_; }
  Symbol<cfg::ParallelDistribution> parallel_distribution() const { return parallel_distribution_; }
  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }

  size_t CalcHashValue() const;

 private:
  std::shared_ptr<const Shape> shape_;
  DataType dtype_;
  Symbol<cfg::ParallelDistribution> parallel_distribution_;
  Symbol<ParallelDesc> parallel_desc_;
};

class ConsistentTensorImpl : public TensorImpl {
 public:
  virtual ~ConsistentTensorImpl() = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return tensor_meta_->shape(); }
  DataType dtype() const override { return tensor_meta_->dtype(); }
  Symbol<cfg::ParallelDistribution> parallel_distribution() const {
    return tensor_meta_->parallel_distribution();
  }
  Symbol<ParallelDesc> parallel_desc() const { return tensor_meta_->parallel_desc(); }
  Symbol<cfg::ParallelDistribution> consumer_forced_parallel_distribution() const {
    return consumer_forced_parallel_distribution_;
  }

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { UNIMPLEMENTED(); }
  void set_dtype(DataType dtype) override { UNIMPLEMENTED(); }
  void set_consumer_forced_parallel_distribution(Symbol<cfg::ParallelDistribution> val) {
    consumer_forced_parallel_distribution_ = val;
  }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<TensorStorage> tensor_storage() const override { OF_UNIMPLEMENTED(); }

 protected:
  ConsistentTensorImpl(Symbol<ConsistentTensorMeta> tensor_meta, bool requires_grad, bool is_leaf)
      : TensorImpl(requires_grad, is_leaf),
        tensor_meta_(tensor_meta),
        consumer_forced_parallel_distribution_() {}

  Symbol<ConsistentTensorMeta> tensor_meta_;
  Symbol<cfg::ParallelDistribution> consumer_forced_parallel_distribution_;
};

class LazyMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyMirroredTensorImpl);
  LazyMirroredTensorImpl(const std::shared_ptr<const Shape>& shape, DataType dtype,
                         const std::shared_ptr<const Device>& device, bool requires_grad,
                         bool is_leaf)
      : MirroredTensorImpl(device, requires_grad, is_leaf), shape_(shape), dtype_(dtype) {}
  ~LazyMirroredTensorImpl() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return shape_; }
  DataType dtype() const override { return dtype_; }
  const std::shared_ptr<const Device>& device() const { return device_; }
  bool is_lazy() const override { return true; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<TensorStorage> tensor_storage() const override { OF_UNIMPLEMENTED(); }

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { shape_ = shape; }
  void set_dtype(DataType dtype) override { dtype_ = dtype; }
  Maybe<void> set_eager_blob_object(
      std::shared_ptr<vm::EagerBlobObject> eager_blob_object) override {
    return Error::Unimplemented();
  }
  Maybe<void> set_tensor_storage(std::shared_ptr<TensorStorage> tensor_storage) override {
    return Error::Unimplemented();
  }

 private:
  std::shared_ptr<const Shape> shape_;
  DataType dtype_;
};

class EagerMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerMirroredTensorImpl);
  EagerMirroredTensorImpl(const std::shared_ptr<vm::EagerBlobObject> eager_blob_object,
                          const std::shared_ptr<const Device>& device, bool requires_grad,
                          bool is_leaf);
  EagerMirroredTensorImpl(const std::shared_ptr<vm::EagerBlobObject> eager_blob_object,
                          const std::shared_ptr<const Device>& device,
                          const std::shared_ptr<TensorStorage> tensor_storage, bool requires_grad,
                          bool is_leaf);
  ~EagerMirroredTensorImpl() override;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override;
  DataType dtype() const override { return dtype_; }
  bool is_lazy() const override { return false; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { return eager_blob_object_; }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override;
  Maybe<TensorStorage> tensor_storage() const override { return tensor_storage_; }

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { UNIMPLEMENTED(); }
  void set_dtype(DataType dtype) override { UNIMPLEMENTED(); }
  Maybe<void> set_tensor_storage(std::shared_ptr<TensorStorage> tensor_storage) override {
    tensor_storage_ = tensor_storage;
    return Maybe<void>::Ok();
  }
  Maybe<void> set_eager_blob_object(
      std::shared_ptr<vm::EagerBlobObject> eager_blob_object) override {
    eager_blob_object_ = eager_blob_object;
    return Maybe<void>::Ok();
  }

 private:
  void Init();

  std::shared_ptr<const Shape> shape_;
  DataType dtype_;
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
