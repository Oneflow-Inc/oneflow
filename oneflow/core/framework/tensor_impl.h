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
#include "oneflow/core/common/shape.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/framework/object.h"
#include "oneflow/core/framework/tensor_arg.h"
#include "oneflow/core/framework/tensor_storage.h"

namespace oneflow {

class MemoryCase;

namespace compatible_py {

class Distribute;
}

class Device;
class DType;

namespace eager {
class EagerBlobObject;
}

namespace one {

class Tensor;
class TensorArg;

class TensorImpl {
 public:
  virtual ~TensorImpl() = default;

  // Getters
  virtual const std::shared_ptr<const Shape>& shape() const = 0;
  virtual const std::shared_ptr<const DType>& dtype() const = 0;
  virtual const std::shared_ptr<const ParallelDesc>& parallel_desc() const = 0;
  virtual bool is_lazy() const = 0;

  // Getters for autograd
  const std::shared_ptr<Tensor>& acc_grad() const { return acc_grad_; }
  const std::shared_ptr<TensorArg>& now_grad_arg() const { return now_grad_arg_; }
  bool requires_grad() const { return requires_grad_; }
  bool is_leaf() const { return is_leaf_; }
  bool retain_grad() const { return retain_grad_; }

  // Setters
  virtual void set_shape(const std::shared_ptr<const Shape>& shape) = 0;
  virtual void set_dtype(const std::shared_ptr<const DType>& dtype) = 0;
  virtual Maybe<void> set_parallel_desc(
      const std::shared_ptr<const ParallelDesc>& parallel_desc) = 0;

  // Setters for autograd
  void set_acc_grad(const std::shared_ptr<Tensor>& grad) { acc_grad_ = grad; }
  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
  void set_retain_grad(bool retain_grad) { retain_grad_ = retain_grad; }
  void set_is_leaf(bool is_leaf) { is_leaf_ = is_leaf; }

  // Getters to be deprecated
  virtual const std::shared_ptr<compatible_py::BlobObject>& blob_object() const = 0;

  // Setters to be deprecated
  virtual Maybe<void> set_blob_object(
      const std::shared_ptr<compatible_py::BlobObject>& blob_object) = 0;

 protected:
  TensorImpl(bool requires_grad, bool is_leaf, bool retain_grad)
      : requires_grad_(requires_grad),
        is_leaf_(is_leaf),
        retain_grad_(retain_grad),
        now_grad_arg_(new TensorArg) {}
  Maybe<void> SyncBlobObject2Attributes(
      const std::shared_ptr<compatible_py::BlobObject>& blob_object);

  // For autograd
  bool requires_grad_;
  bool is_leaf_;
  bool retain_grad_;
  std::shared_ptr<Tensor> acc_grad_;
  std::shared_ptr<TensorArg> now_grad_arg_;
};

class MirroredTensorImpl : public TensorImpl {
 public:
  virtual ~MirroredTensorImpl() = default;

  // Getters
  const std::shared_ptr<const Device>& device() const { return device_; }
  const std::shared_ptr<const ParallelDesc>& parallel_desc() const override {
    return parallel_desc_;
  }

  // Setters
  Maybe<void> set_parallel_desc(const std::shared_ptr<const ParallelDesc>& parallel_desc) override;
  Maybe<void> set_device(const std::shared_ptr<const Device>& device);

 protected:
  MirroredTensorImpl(const std::shared_ptr<const Device>& device, bool requires_grad, bool is_leaf,
                     bool retain_grad)
      : TensorImpl(requires_grad, is_leaf, retain_grad) {
    set_device(device);
  }

  std::shared_ptr<const Device> device_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;
};

class ConsistentTensorImpl : public TensorImpl {
 public:
  virtual ~ConsistentTensorImpl() = default;

  // Getters
  const std::shared_ptr<const Device>& device() const { return device_ /* always nullptr*/; }
  const std::shared_ptr<const ParallelDesc>& parallel_desc() const override {
    return parallel_desc_;
  };
  virtual const std::shared_ptr<const compatible_py::Distribute>& distribute() const = 0;

  // Setters
  Maybe<void> set_parallel_desc(const std::shared_ptr<const ParallelDesc>& parallel_desc) override;

  virtual void set_distribute(
      const std::shared_ptr<const compatible_py::Distribute>& distribute) = 0;

 protected:
  ConsistentTensorImpl(const std::shared_ptr<const ParallelDesc>& parallel_desc, bool requires_grad,
                       bool is_leaf, bool retain_grad)
      : TensorImpl(requires_grad, is_leaf, retain_grad), parallel_desc_(parallel_desc) {}

  const std::shared_ptr<const Device> device_;  // always nullptr
  std::shared_ptr<const ParallelDesc> parallel_desc_;
};

class LazyMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyMirroredTensorImpl);
  LazyMirroredTensorImpl(const std::shared_ptr<const Shape>& shape,
                         const std::shared_ptr<const DType>& dtype,
                         const std::shared_ptr<const Device>& device, bool requires_grad,
                         bool is_leaf, bool retain_grad)
      : MirroredTensorImpl(device, requires_grad, is_leaf, retain_grad),
        shape_(shape),
        dtype_(dtype) {}
  ~LazyMirroredTensorImpl() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return shape_; }
  const std::shared_ptr<const DType>& dtype() const override { return dtype_; }
  const std::shared_ptr<const Device>& device() const { return device_; }
  bool is_lazy() const override { return true; }

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { shape_ = shape; }
  void set_dtype(const std::shared_ptr<const DType>& dtype) override { dtype_ = dtype; }

  // Getters to be deprecated
  const std::shared_ptr<compatible_py::BlobObject>& blob_object() const override {
    UNIMPLEMENTED();
  }

  // Setters to be deprecated
  Maybe<void> set_blob_object(
      const std::shared_ptr<compatible_py::BlobObject>& blob_object) override {
    UNIMPLEMENTED_THEN_RETURN();
  }

 private:
  std::shared_ptr<const Shape> shape_;
  std::shared_ptr<const DType> dtype_;
};

class EagerMirroredTensorImpl final : public MirroredTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerMirroredTensorImpl);
  EagerMirroredTensorImpl(const std::shared_ptr<const Shape>& shape,
                          const std::shared_ptr<const DType>& dtype,
                          const std::shared_ptr<const Device>& device, bool requires_grad,
                          bool is_leaf, bool retain_grad)
      : EagerMirroredTensorImpl(shape, dtype, device, std::make_shared<TensorStorage>(),
                                requires_grad, is_leaf, retain_grad) {}
  EagerMirroredTensorImpl(const std::shared_ptr<const Shape>& shape,
                          const std::shared_ptr<const DType>& dtype,
                          const std::shared_ptr<const Device>& device,
                          const std::shared_ptr<TensorStorage>& tensor_storage, bool requires_grad,
                          bool is_leaf, bool retain_grad)
      : MirroredTensorImpl(device, requires_grad, is_leaf, retain_grad),
        shape_(shape),
        dtype_(dtype),
        tensor_storage_(tensor_storage),
        blob_object_id_(Error::ValueError("blob_object_id is not initialized")) {}
  ~EagerMirroredTensorImpl() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return shape_; }
  const std::shared_ptr<const DType>& dtype() const override { return dtype_; }
  bool is_lazy() const override { return false; }
  const std::shared_ptr<eager::EagerBlobObject>& eager_blob_object() const {
    return eager_blob_object_;
  }
  Maybe<int64_t> blob_object_id() const { return blob_object_id_; }

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { shape_ = shape; }
  void set_dtype(const std::shared_ptr<const DType>& dtype) override { dtype_ = dtype; }
  void set_blob_object_id(int64_t blob_object_id) { blob_object_id_ = blob_object_id; }
  TensorStorage* mut_tensor_storage() { return tensor_storage_.get(); }

  // Getters to be deprecated
  const std::shared_ptr<compatible_py::BlobObject>& blob_object() const override {
    return blob_object_;
  }

  // Setters to be deprecated
  Maybe<void> set_blob_object(
      const std::shared_ptr<compatible_py::BlobObject>& blob_object) override;

  // other methods
  Maybe<void> InitEagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case);

 private:
  std::shared_ptr<const Shape> shape_;
  std::shared_ptr<const DType> dtype_;
  std::shared_ptr<compatible_py::BlobObject> blob_object_;
  std::shared_ptr<TensorStorage> tensor_storage_;
  std::shared_ptr<eager::EagerBlobObject> eager_blob_object_;
  Maybe<int64_t> blob_object_id_;
};

class LazyConsistentTensorImpl final : public ConsistentTensorImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyConsistentTensorImpl);
  LazyConsistentTensorImpl(const std::shared_ptr<const Shape>& shape,
                           const std::shared_ptr<const DType>& dtype,
                           const std::shared_ptr<const compatible_py::Distribute>& distribute,
                           const std::shared_ptr<const ParallelDesc>& parallel_desc,
                           bool requires_grad, bool is_leaf, bool retain_grad)
      : ConsistentTensorImpl(parallel_desc, requires_grad, is_leaf, retain_grad),
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

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { shape_ = shape; }
  void set_dtype(const std::shared_ptr<const DType>& dtype) override { dtype_ = dtype; }
  void set_distribute(const std::shared_ptr<const compatible_py::Distribute>& distribute) override {
    distribute_ = distribute;
  }

  // Getters to be deprecated
  const std::shared_ptr<compatible_py::BlobObject>& blob_object() const override {
    UNIMPLEMENTED();
  }

  // Setters to be deprecated
  Maybe<void> set_blob_object(
      const std::shared_ptr<compatible_py::BlobObject>& blob_object) override {
    UNIMPLEMENTED_THEN_RETURN();
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
                            bool requires_grad, bool is_leaf, bool retain_grad)
      : ConsistentTensorImpl(parallel_desc, requires_grad, is_leaf, retain_grad),
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

  // Setters
  void set_shape(const std::shared_ptr<const Shape>& shape) override { shape_ = shape; }
  void set_dtype(const std::shared_ptr<const DType>& dtype) override { dtype_ = dtype; }
  void set_distribute(const std::shared_ptr<const compatible_py::Distribute>& distribute) override {
    distribute_ = distribute;
  }

  // Getters to be deprecated
  const std::shared_ptr<compatible_py::BlobObject>& blob_object() const override {
    return blob_object_;
  }

  // Setters to be deprecated
  Maybe<void> set_blob_object(
      const std::shared_ptr<compatible_py::BlobObject>& blob_object) override;

 private:
  std::shared_ptr<const Shape> shape_;
  std::shared_ptr<const DType> dtype_;
  std::shared_ptr<const compatible_py::Distribute> distribute_;
  std::shared_ptr<compatible_py::BlobObject> blob_object_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_
