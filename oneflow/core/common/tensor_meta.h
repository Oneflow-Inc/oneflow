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
  TensorMeta(DataType dtype) : data_type_(dtype), is_dynamic_(false) {}
  TensorMeta(const TensorMeta& other) = default;
  TensorMeta(TensorMeta&&) = default;
  virtual ~TensorMeta() = default;

  virtual const std::shared_ptr<const Shape>& shape_ptr() const = 0;
  virtual const std::shared_ptr<const Stride>& stride_ptr() const = 0;
  virtual bool is_contiguous() const = 0;

  DataType dtype() const { return data_type_; }
  DataType data_type() const override { return data_type_; }
  bool is_dynamic() const override { return is_dynamic_; }

  virtual void set_shape(const Shape& shape) override { PRINT_BUG_PROMPT_AND_ABORT(); }
  virtual void set_stride(const Stride& stride) override { PRINT_BUG_PROMPT_AND_ABORT(); }
  virtual void set_data_type(DataType data_type) override { PRINT_BUG_PROMPT_AND_ABORT(); }
  virtual void set_is_dynamic(bool is_dynamic) override { PRINT_BUG_PROMPT_AND_ABORT(); }

 protected:
  DataType data_type_;
  bool is_dynamic_;
};

class MutTensorMeta : public TensorMeta {
 public:
  // uninitialized MutTensorMeta.
  MutTensorMeta();
  MutTensorMeta(const MutTensorMeta& other)
      : TensorMeta(other),
        shape_(std::make_shared<const Shape>(*other.shape_)),
        stride_(std::make_shared<const Stride>(*other.stride_)) {}
  MutTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype);
  MutTensorMeta(const std::shared_ptr<const Shape>& shape,
                const std::shared_ptr<const Stride>& stride, DataType dtype);
  MutTensorMeta(const Shape& shape, DataType dtype);
  MutTensorMeta(const Shape& shape, const Stride& stride, DataType dtype);
  virtual ~MutTensorMeta() = default;

  const std::shared_ptr<const Shape>& shape_ptr() const override { return shape_; }
  const std::shared_ptr<const Stride>& stride_ptr() const override { return stride_; }
  const Shape& shape() const override { return *shape_; }
  const Stride& stride() const override { return *stride_; }
  bool is_contiguous() const override { return IsContiguous(*shape_, *stride_); }

  void set_shape(const Shape& shape) override { *const_cast<Shape*>(shape_.get()) = shape; }
  void set_stride(const Stride& stride) override { *const_cast<Stride*>(stride_.get()) = stride; }
  void set_data_type(DataType data_type) override { data_type_ = data_type; }
  void set_is_dynamic(bool is_dynamic) override { is_dynamic_ = is_dynamic; }

  bool operator==(const MutTensorMeta& other) const;
  size_t CalcHashValue() const;

  MutTensorMeta& operator=(const MutTensorMeta& other) {
    this->data_type_ = other.data_type_;
    this->is_dynamic_ = other.is_dynamic_;
    this->shape_ = std::make_shared<const Shape>(*other.shape_);
    this->stride_ = std::make_shared<const Stride>(*other.stride_);
    return *this;
  }

 protected:
  std::shared_ptr<const Shape> shape_;
  std::shared_ptr<const Stride> stride_;
};

class ConstTensorMeta : public TensorMeta {
 public:
  // uninitialized ConstTensorMeta.
  ConstTensorMeta();
  ConstTensorMeta(const ConstTensorMeta&) = default;
  ConstTensorMeta(Symbol<Shape> shape, DataType dtype);
  ConstTensorMeta(Symbol<Shape> shape, Symbol<Stride> stride, DataType dtype);
  ConstTensorMeta(const Shape& shape, DataType dtype) : ConstTensorMeta(SymbolOf(shape), dtype) {}
  ConstTensorMeta(const Shape& shape, const Stride& stride, DataType dtype)
      : ConstTensorMeta(SymbolOf(shape), SymbolOf(stride), dtype) {}
  virtual ~ConstTensorMeta() = default;

  const std::shared_ptr<const Shape>& shape_ptr() const override {
    return shape_.shared_from_symbol();
  }
  const std::shared_ptr<const Stride>& stride_ptr() const override {
    return stride_.shared_from_symbol();
  }
  const Shape& shape() const override { return *shape_; }
  const Stride& stride() const override { return *stride_; }
  bool is_contiguous() const override { return IsContiguous(*shape_, *stride_); }

  bool operator==(const ConstTensorMeta& other) const;
  size_t CalcHashValue() const;

  ConstTensorMeta& operator=(const ConstTensorMeta& other) {
    this->data_type_ = other.data_type_;
    this->is_dynamic_ = other.is_dynamic_;
    this->shape_ = other.shape_;
    this->stride_ = other.stride_;
    return *this;
  }

 protected:
  Symbol<Shape> shape_;
  Symbol<Stride> stride_;
};

class LocalTensorMeta : public ConstTensorMeta {
 public:
  // uninitialized LocalTensorMeta.
  LocalTensorMeta();
  LocalTensorMeta(const LocalTensorMeta&) = default;
  LocalTensorMeta(Symbol<Shape> shape, DataType dtype, Symbol<Device> device);
  LocalTensorMeta(Symbol<Shape> shape, Symbol<Stride> stride, DataType dtype,
                  Symbol<Device> device);
  LocalTensorMeta(const Shape& shape, DataType dtype, Symbol<Device> device)
      : LocalTensorMeta(SymbolOf(shape), dtype, device) {}
  LocalTensorMeta(const Shape& shape, const Stride& stride, DataType dtype, Symbol<Device> device)
      : LocalTensorMeta(SymbolOf(shape), SymbolOf(stride), dtype, device) {}
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
  MutLocalTensorMeta(const Shape& shape, DataType dtype, Symbol<Device> device);
  MutLocalTensorMeta(const Shape& shape, const Stride& stride, DataType dtype,
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

class GlobalTensorMeta : public ConstTensorMeta {
 public:
  GlobalTensorMeta(Symbol<Shape> shape, DataType dtype, Symbol<NdSbp> nd_sbp,
                   Symbol<ParallelDesc> parallel_desc)
      : ConstTensorMeta(shape, dtype), nd_sbp_(nd_sbp), parallel_desc_(parallel_desc) {}
  GlobalTensorMeta(const Shape& shape, DataType dtype, Symbol<NdSbp> nd_sbp,
                   Symbol<ParallelDesc> parallel_desc)
      : GlobalTensorMeta(SymbolOf(shape), dtype, nd_sbp, parallel_desc) {}
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
