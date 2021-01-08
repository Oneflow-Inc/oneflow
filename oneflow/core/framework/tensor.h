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
#include "oneflow/core/register/blob.h"

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
  virtual cfg::DataType dtype() const = 0;
  virtual std::shared_ptr<cfg::ParallelConf> parallel_conf() const = 0;
};

namespace one {

class FunctionNode;

class TensorImpl {
 public:
  TensorImpl() = default;
  TensorImpl(const std::shared_ptr<Shape>& shape, cfg::DataType dtype,
             const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : shape_(shape), dtype_(dtype), parallel_conf_(parallel_conf) {}
  ~TensorImpl() = default;

  std::shared_ptr<Blob> storage() const { return storage_; }
  std::shared_ptr<Shape> shape() const { return shape_; }
  cfg::DataType dtype() const { return dtype_; }
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const { return parallel_conf_; }
  int32_t dim() const { return shape_->NumAxes(); }

  bool has_storage() const { return static_cast<bool>(storage_); }

  template<typename T = void>
  const T* data() const {
    CHECK(has_storage());
    return storage_->dptr<T>();
  }

  template<typename T = void>
  T* mutable_data() {
    CHECK(has_storage());
    return storage_->mut_dptr<T>();
  }

 private:
  std::shared_ptr<Blob> storage_;
  std::shared_ptr<Shape> shape_;
  cfg::DataType dtype_;
  std::shared_ptr<cfg::ParallelConf> parallel_conf_;
  // TODO: Strides related features will be supported later
};

// one::Tensor will replace oneflow::Tensor in the future
class Tensor : public oneflow::Tensor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Tensor);
  Tensor() = delete;
  Tensor(const std::shared_ptr<Shape>& sizes, cfg::DataType dtype,
         const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
    impl_ = std::make_shared<TensorImpl>(sizes, dtype, parallel_conf);
  }
  ~Tensor() override = default;

  // Basic Properties
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override {
    return impl_->parallel_conf();
  }
  std::shared_ptr<Shape> shape() const override { return impl_->shape(); }
  cfg::DataType dtype() const override { return impl_->dtype(); }
  bool defined() const { return static_cast<bool>(impl_); }
  bool has_storage() const { return defined() && impl_->has_storage(); }
  bool requires_grad() const { return is_requires_grad_; }
  bool is_leaf() const { return is_leaf_; }
  std::shared_ptr<Tensor> clone() const;  // malloc memory in same place and have same grad_function
  std::shared_ptr<Tensor> detach() const;  // share blob but have no grad_function
  int32_t dim() const { return impl_->dim(); }
  std::shared_ptr<Tensor> grad() const { return grad_; }

  // Inherit some virtual unimplement interface
  std::shared_ptr<cfg::LogicalBlobId> lbi() const override { UNIMPLEMENTED(); }
  std::string logical_blob_name() const override { UNIMPLEMENTED(); }
  std::string op_name() const override { UNIMPLEMENTED(); }
  std::string blob_name() const override { UNIMPLEMENTED(); }

  // Autograd
  void Backward(const std::shared_ptr<Tensor>& grad, bool retain_graph = false) { UNIMPLEMENTED(); }
  void SetFuncNode(const std::shared_ptr<FunctionNode>& func_ptr) { grad_func_ = func_ptr; }

 protected:
  std::shared_ptr<TensorImpl> impl_;
  std::weak_ptr<FunctionNode> grad_func_;
  std::shared_ptr<Tensor> grad_;
  bool is_requires_grad_ = false;
  bool is_leaf_ = false;
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
