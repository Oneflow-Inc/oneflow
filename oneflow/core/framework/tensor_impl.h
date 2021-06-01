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

namespace oneflow {

class MemoryCase;
class VmLocalDepObject;

namespace compatible_py {

class Distribute;
}

class Shape;
class Device;
class DType;

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
  virtual const std::shared_ptr<const DType>& dtype() const = 0;
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

  // Setters
  virtual void set_shape(const std::shared_ptr<const Shape>& shape) = 0;
  virtual void set_dtype(const std::shared_ptr<const DType>& dtype) = 0;

  // Setters for autograd
  void set_acc_grad(const std::shared_ptr<Tensor>& grad) { autograd_meta_->set_acc_grad(grad); }
  std::shared_ptr<Tensor> mut_acc_grad() { return autograd_meta_->mut_acc_grad(); }
  void set_requires_grad(bool requires_grad) { autograd_meta_->set_requires_grad(requires_grad); }
  void set_retain_grad(bool retain_grad) { autograd_meta_->set_retain_grad(retain_grad); }
  void set_is_leaf(bool is_leaf) { autograd_meta_->set_is_leaf(is_leaf); }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() { return autograd_meta_; }

 protected:
  TensorImpl(bool requires_grad, bool is_leaf)
      : autograd_meta_(new AutogradMeta(requires_grad, is_leaf)) {}

 protected:
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

 protected:
  MirroredTensorImpl(const std::shared_ptr<const Device>& device, bool requires_grad, bool is_leaf)
      : TensorImpl(requires_grad, is_leaf) {
    set_device(device);
  }

  std::shared_ptr<const Device> device_;
};

class ConsistentTensorImpl : public TensorImpl {
 public:
  virtual ~ConsistentTensorImpl() = default;

  // Getters
  const std::shared_ptr<const ParallelDesc>& parallel_desc() const { return parallel_desc_; };
  virtual const std::shared_ptr<const compatible_py::Distribute>& distribute() const = 0;

  // Setters
  Maybe<void> set_parallel_desc(const std::shared_ptr<const ParallelDesc>& parallel_desc);

  virtual void set_distribute(
      const std::shared_ptr<const compatible_py::Distribute>& distribute) = 0;

 protected:
  ConsistentTensorImpl(const std::shared_ptr<const ParallelDesc>& parallel_desc, bool requires_grad,
                       bool is_leaf)
      : TensorImpl(requires_grad, is_leaf), parallel_desc_(parallel_desc) {}

  std::shared_ptr<const ParallelDesc> parallel_desc_;
};

class LazyMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyMirroredTensorImpl);
  LazyMirroredTensorImpl(const std::shared_ptr<const Shape>& shape,
                         const std::shared_ptr<const DType>& dtype,
                         const std::shared_ptr<const Device>& device, bool requires_grad,
                         bool is_leaf)
      : MirroredTensorImpl(device, requires_grad, is_leaf), shape_(shape), dtype_(dtype) {}
  ~LazyMirroredTensorImpl() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return shape_; }
  const std::shared_ptr<const DType>& dtype() const override { return dtype_; }
  const std::shared_ptr<const Device>& device() const { return device_; }
  bool is_lazy() const override { return true; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override { OF_UNIMPLEMENTED(); }

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { shape_ = shape; }
  void set_dtype(const std::shared_ptr<const DType>& dtype) override { dtype_ = dtype; }
  Maybe<void> set_eager_blob_object(
      std::shared_ptr<vm::EagerBlobObject> eager_blob_object) override {
    return Error::Unimplemented();
  }

 private:
  std::shared_ptr<const Shape> shape_;
  std::shared_ptr<const DType> dtype_;
};

class EagerMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerMirroredTensorImpl);
  EagerMirroredTensorImpl(const std::shared_ptr<vm::EagerBlobObject> eager_blob_object,
                          const std::shared_ptr<const Device>& device, bool requires_grad,
                          bool is_leaf);
  EagerMirroredTensorImpl(const std::shared_ptr<const Shape>& shape,
                          const std::shared_ptr<const DType>& dtype,
                          const std::shared_ptr<const Device>& device,
                          const std::shared_ptr<TensorStorage>& tensor_storage, bool requires_grad,
                          bool is_leaf);
  ~EagerMirroredTensorImpl() override;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override;
  const std::shared_ptr<const DType>& dtype() const override { return dtype_; }
  bool is_lazy() const override { return false; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { return eager_blob_object_; }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override;

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { UNIMPLEMENTED(); }
  void set_dtype(const std::shared_ptr<const DType>& dtype) override { UNIMPLEMENTED(); }
  TensorStorage* mut_tensor_storage() { return tensor_storage_.get(); }
  Maybe<void> set_eager_blob_object(
      std::shared_ptr<vm::EagerBlobObject> eager_blob_object) override {
    eager_blob_object_ = eager_blob_object;
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<const Shape> shape_;
  std::shared_ptr<const DType> dtype_;
  std::shared_ptr<TensorStorage> tensor_storage_;
  std::shared_ptr<vm::EagerBlobObject> eager_blob_object_;
};

class LazyConsistentTensorImpl final : public ConsistentTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyConsistentTensorImpl);
  LazyConsistentTensorImpl(const std::shared_ptr<const Shape>& shape,
                           const std::shared_ptr<const DType>& dtype,
                           const std::shared_ptr<const compatible_py::Distribute>& distribute,
                           const std::shared_ptr<const ParallelDesc>& parallel_desc,
                           bool requires_grad, bool is_leaf)
      : ConsistentTensorImpl(parallel_desc, requires_grad, is_leaf),
        shape_(shape),
        dtype_(dtype),
        distribute_(distribute) {}
  ~LazyConsistentTensorImpl() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return shape_; }
  const std::shared_ptr<const DType>& dtype() const override { return dtype_; }
  const std::shared_ptr<const compatible_py::Distribute>& distribute() const override {
    return distribute_;
  }
  bool is_lazy() const override { return true; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override { OF_UNIMPLEMENTED(); }

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { shape_ = shape; }
  void set_dtype(const std::shared_ptr<const DType>& dtype) override { dtype_ = dtype; }
  void set_distribute(const std::shared_ptr<const compatible_py::Distribute>& distribute) override {
    distribute_ = distribute;
  }

 private:
  std::shared_ptr<const Shape> shape_;
  std::shared_ptr<const DType> dtype_;
  std::shared_ptr<const compatible_py::Distribute> distribute_;
};

class EagerConsistentTensorImpl final : public ConsistentTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerConsistentTensorImpl);
  EagerConsistentTensorImpl(const std::shared_ptr<const Shape>& shape,
                            const std::shared_ptr<const DType>& dtype,
                            const std::shared_ptr<const compatible_py::Distribute>& distribute,
                            const std::shared_ptr<const ParallelDesc>& parallel_desc,
                            bool requires_grad, bool is_leaf)
      : ConsistentTensorImpl(parallel_desc, requires_grad, is_leaf),
        shape_(shape),
        dtype_(dtype),
        distribute_(distribute) {}
  ~EagerConsistentTensorImpl() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return shape_; }
  const std::shared_ptr<const DType>& dtype() const override { return dtype_; }
  const std::shared_ptr<const compatible_py::Distribute>& distribute() const override {
    return distribute_;
  }
  bool is_lazy() const override { return false; }

  // Getters valid only for EagerMirroredTensorImpl
  Maybe<vm::EagerBlobObject> eager_blob_object() const override { OF_UNIMPLEMENTED(); }
  Maybe<VmLocalDepObject> compute_local_dep_object() const override { OF_UNIMPLEMENTED(); }

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { shape_ = shape; }
  void set_dtype(const std::shared_ptr<const DType>& dtype) override { dtype_ = dtype; }
  void set_distribute(const std::shared_ptr<const compatible_py::Distribute>& distribute) override {
    distribute_ = distribute;
  }

 private:
  std::shared_ptr<const Shape> shape_;
  std::shared_ptr<const DType> dtype_;
  std::shared_ptr<const compatible_py::Distribute> distribute_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_
