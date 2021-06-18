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

class Blob;

namespace cfg {

class LogicalBlobId;
class ParallelConf;

}  // namespace cfg

class Tensor {
 public:
  virtual ~Tensor() = default;

  virtual std::shared_ptr<cfg::LogicalBlobId> lbi() const = 0;
  virtual std::string logical_blob_name() const = 0;
  virtual std::string op_name() const = 0;
  virtual std::string blob_name() const = 0;
  virtual std::shared_ptr<Shape> shape() const = 0;
  virtual DataType dtype() const = 0;
  virtual std::shared_ptr<cfg::ParallelConf> parallel_conf() const = 0;
};

namespace cfg {
class ParallelDistribution;
}
class Device;

namespace one {

class FunctionNode;

class Tensor {
 public:
  virtual ~Tensor() = default;

  // Getters
  virtual const std::shared_ptr<const Shape>& shape() const = 0;
  virtual DataType dtype() const = 0;
  virtual Maybe<Symbol<cfg::ParallelDistribution>> parallel_distribution() const = 0;
  virtual Maybe<Symbol<ParallelDesc>> parallel_desc() const = 0;
  virtual Maybe<const Device> device() const = 0;
  virtual Maybe<std::shared_ptr<const Device>*> mut_device() { OF_UNIMPLEMENTED(); }
  virtual bool is_consistent() const = 0;
  virtual bool is_lazy() const = 0;
  virtual Maybe<Symbol<cfg::ParallelDistribution>> consumer_forced_parallel_distribution()
      const = 0;
  virtual const TensorMeta& tensor_meta() const = 0;

  // Getters valid only for EagerMirroredTensor
  virtual Maybe<vm::EagerBlobObject> eager_blob_object() const = 0;
  virtual Maybe<VmLocalDepObject> compute_local_dep_object() const = 0;
  virtual Maybe<TensorStorage> tensor_storage() const = 0;

  // Setters
  virtual Maybe<void> set_consumer_forced_parallel_distribution(
      Symbol<cfg::ParallelDistribution> val) = 0;

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

class ConsistentTensor;
class MirroredTensor;

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
  Maybe<const Device> device() const override { return impl_->device(); }
  Maybe<std::shared_ptr<const Device>*> mut_device() override { return impl_->mut_device(); }
  bool is_lazy() const override { return impl_->is_lazy(); }
  bool is_consistent() const override { return false; }
  Maybe<Symbol<cfg::ParallelDistribution>> consumer_forced_parallel_distribution() const override {
    OF_UNIMPLEMENTED();
  }
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

  // Setters
  Maybe<void> set_consumer_forced_parallel_distribution(
      Symbol<cfg::ParallelDistribution> val) override {
    OF_UNIMPLEMENTED();
  }

  Maybe<void> set_tensor_storage(const std::shared_ptr<TensorStorage>& tensor_storage) {
    return impl_->set_tensor_storage(tensor_storage);
  }

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
                                          const std::shared_ptr<const Device>& device, bool is_lazy,
                                          bool requires_grad, bool is_leaf);

  MirroredTensorImpl* mut_impl() { return impl_.get(); }
  user_op::TensorDesc* mut_tensor_meta() override { return impl_->mut_tensor_meta(); }

  static std::shared_ptr<MirroredTensor> MakeEagerTensor(
      const std::shared_ptr<vm::EagerBlobObject> eager_blob_object,
      const std::shared_ptr<const Device>& device,
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
  Maybe<const Device> device() const override { OF_UNIMPLEMENTED(); }
  bool is_lazy() const override { return impl_->is_lazy(); }
  bool is_consistent() const override { return true; }
  Maybe<Symbol<cfg::ParallelDistribution>> consumer_forced_parallel_distribution() const override {
    return impl_->consumer_forced_parallel_distribution();
  }
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
  Maybe<TensorStorage> tensor_storage() const override { return impl_->tensor_storage(); }

  // Setters
  Maybe<void> set_consumer_forced_parallel_distribution(
      Symbol<cfg::ParallelDistribution> val) override {
    impl_->set_consumer_forced_parallel_distribution(val);
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

  user_op::TensorDesc* mut_tensor_meta() override { return impl_->mut_tensor_meta(); }

 private:
  std::shared_ptr<ConsistentTensorImpl> impl_;
};

}  // namespace one

namespace user_op {

class Tensor {
 public:
  ~Tensor() = default;

  virtual const ShapeView& shape() const = 0;
  virtual MutShapeView* mut_shape() = 0;
  virtual DataType data_type() const = 0;
  virtual const MemoryCase& mem_case() const = 0;
  virtual const void* raw_dptr() const = 0;
  virtual void* mut_raw_dptr() = 0;

  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>();
    return reinterpret_cast<const T*>(raw_dptr());
  }

  template<typename T = void>
  T* mut_dptr() {
    CheckDataType<T>();
    return reinterpret_cast<T*>(mut_raw_dptr());
  }

 protected:
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
                   && data_type() != DataType::kChar && data_type() != GetDataType<T>::value))
        << "tensor data_type mismatched. value: " << DataType_Name(data_type())
        << ", template T:" << DataType_Name(GetDataType<T>::value);
  }
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
