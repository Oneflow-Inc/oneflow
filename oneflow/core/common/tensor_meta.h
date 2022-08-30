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
#ifndef ONEFLOW_COMMON_TENSOR_META_H_
#define ONEFLOW_COMMON_TENSOR_META_H_

#include <memory>
#include "oneflow/core/common/tensor_desc.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class NdSbp;
class Shape;
class Stride;
class Device;
class ParallelDesc;

namespace one {

bool IsContiguous(const Shape& shape, const Stride& stride);
bool IsContiguous(const ShapeView& shape_view, const Stride& stride);

class TensorMeta : public user_op::TensorDesc {
 public:
  TensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype)
      : shape_(shape),
        stride_(std::shared_ptr<Stride>(new Stride(*shape))),
        data_type_(dtype),
        is_dynamic_(false) {}
  TensorMeta(const std::shared_ptr<const Shape>& shape, const std::shared_ptr<const Stride>& stride,
             DataType dtype)
      : shape_(shape), stride_(stride), data_type_(dtype), is_dynamic_(false) {}
  TensorMeta(const TensorMeta& other)
      : shape_(std::make_shared<Shape>(*other.shape_)),
        stride_(std::make_shared<Stride>(*other.stride_)),
        data_type_(other.data_type_),
        is_dynamic_(other.is_dynamic_) {}
  TensorMeta(TensorMeta&&) = default;
  virtual ~TensorMeta() = default;

  const std::shared_ptr<const Shape>& shape_ptr() const { return shape_; }
  const std::shared_ptr<const Stride>& stride_ptr() const { return stride_; }

  const Shape& shape() const override { return *shape_; }
  const Stride& stride() const override { return *stride_; }
  DataType dtype() const { return data_type_; }
  DataType data_type() const override { return data_type_; }
  bool is_dynamic() const override { return is_dynamic_; }
  bool is_contiguous() const { return IsContiguous(shape(), *stride_); }

  virtual Shape* mut_shape() override {
    PRINT_BUG_PROMPT_AND_ABORT();
    return nullptr;
  }
  virtual Stride* mut_stride() override {
    PRINT_BUG_PROMPT_AND_ABORT();
    return nullptr;
  }
  virtual DataType* mut_data_type() override {
    PRINT_BUG_PROMPT_AND_ABORT();
    return nullptr;
  }
  virtual bool* mut_is_dynamic() override {
    PRINT_BUG_PROMPT_AND_ABORT();
    return nullptr;
  }
  virtual void set_is_dynamic(bool val) override { PRINT_BUG_PROMPT_AND_ABORT(); }

 protected:
  TensorMeta& operator=(const TensorMeta& other) {
    this->shape_ = std::make_shared<const Shape>(*other.shape_);
    this->stride_ = std::make_shared<const Stride>(*other.stride_);
    this->data_type_ = other.data_type_;
    this->is_dynamic_ = other.is_dynamic_;
    return *this;
  }

  std::shared_ptr<const Shape> shape_;
  std::shared_ptr<const Stride> stride_;
  DataType data_type_;
  bool is_dynamic_;
};

class MutTensorMeta : public TensorMeta {
 public:
  // uninitialized MutTensorMeta.
  MutTensorMeta();
  MutTensorMeta(const MutTensorMeta&) = default;
  MutTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype);
  MutTensorMeta(const std::shared_ptr<const Shape>& shape,
                const std::shared_ptr<const Stride>& stride, DataType dtype);
  virtual ~MutTensorMeta() = default;

  Shape* mut_shape() override { return const_cast<Shape*>(shape_.get()); }
  Stride* mut_stride() override { return const_cast<Stride*>(stride_.get()); }
  DataType* mut_data_type() override { return &data_type_; }
  bool* mut_is_dynamic() override { return &is_dynamic_; }
  void set_is_dynamic(bool val) override { is_dynamic_ = val; }

  void set_shape(const std::shared_ptr<const Shape>& val) { shape_ = val; }
  void set_stride(const std::shared_ptr<const Stride>& val) { stride_ = val; }
  DataType* mut_dtype() { return &data_type_; }
  void set_dtype(DataType data_type) { data_type_ = data_type; }

  bool operator==(const MutTensorMeta& other) const;
  size_t CalcHashValue() const;

  MutTensorMeta& operator=(const MutTensorMeta& other) = default;
};

class LocalTensorMeta : public TensorMeta {
 public:
  // uninitialized LocalTensorMeta.
  LocalTensorMeta();
  LocalTensorMeta(const LocalTensorMeta&) = default;
  LocalTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype, Symbol<Device> device);
  LocalTensorMeta(const std::shared_ptr<const Shape>& shape,
                  const std::shared_ptr<const Stride>& stride, DataType dtype,
                  Symbol<Device> device);
  virtual ~LocalTensorMeta() = default;

  const Symbol<Device>& device() const { return device_; }

  bool operator==(const LocalTensorMeta& other) const;
  size_t CalcHashValue() const;

  LocalTensorMeta& operator=(const LocalTensorMeta& other) = default;

 private:
  Symbol<Device> device_;
};

class MutLocalTensorMeta : public MutTensorMeta {
 public:
  // uninitialized MutLocalTensorMeta.
  MutLocalTensorMeta();
  MutLocalTensorMeta(const MutLocalTensorMeta&) = default;
  MutLocalTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                     Symbol<Device> device);
  MutLocalTensorMeta(const std::shared_ptr<const Shape>& shape,
                     const std::shared_ptr<const Stride>& stride, DataType dtype,
                     Symbol<Device> device);
  virtual ~MutLocalTensorMeta() = default;

  const Symbol<Device>& device() const { return device_; }

  Symbol<Device>* mut_device() { return &device_; }

  bool operator==(const MutLocalTensorMeta& other) const;
  size_t CalcHashValue() const;

  MutLocalTensorMeta& operator=(const MutLocalTensorMeta& other) = default;

 private:
  Symbol<Device> device_;
};

class GlobalTensorMeta : public TensorMeta {
 public:
  GlobalTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype, Symbol<NdSbp> nd_sbp,
                   Symbol<ParallelDesc> parallel_desc)
      : TensorMeta(shape, dtype), nd_sbp_(nd_sbp), parallel_desc_(parallel_desc) {}
  GlobalTensorMeta(const GlobalTensorMeta&) = default;
  GlobalTensorMeta(GlobalTensorMeta&&) = default;
  virtual ~GlobalTensorMeta() = default;

  bool operator==(const GlobalTensorMeta& other) const;

  Symbol<NdSbp> nd_sbp() const { return nd_sbp_; }
  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }

  size_t CalcHashValue() const;

 private:
  Symbol<NdSbp> nd_sbp_;
  Symbol<ParallelDesc> parallel_desc_;
};

}  // namespace one
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::one::LocalTensorMeta> final {
  size_t operator()(const oneflow::one::LocalTensorMeta& local_tensor_meta) const {
    return local_tensor_meta.CalcHashValue();
  }
};

template<>
struct hash<oneflow::one::GlobalTensorMeta> final {
  size_t operator()(const oneflow::one::GlobalTensorMeta& global_tensor_meta) const {
    return global_tensor_meta.CalcHashValue();
  }
};

}  // namespace std

#endif  // ONEFLOW_COMMON_TENSOR_META_H_
