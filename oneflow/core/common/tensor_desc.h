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
#ifndef ONEFLOW_CORE_COMMON_TENSOR_DESC_H_
#define ONEFLOW_CORE_COMMON_TENSOR_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/data_type.pb.h"

namespace oneflow {

class BlobDescProto;

namespace user_op {

class TensorDesc {
 public:
  virtual ~TensorDesc() = default;
  TensorDesc& operator=(const TensorDesc& rhs);
  bool operator==(const TensorDesc&) const;

  virtual const Shape& shape() const = 0;
  virtual void set_shape(const Shape& shape) = 0;
  virtual const Stride& stride() const = 0;
  virtual void set_stride(const Stride& stride) = 0;
  virtual DataType data_type() const = 0;
  virtual void set_data_type(DataType data_type) = 0;

  virtual bool is_dynamic() const = 0;
  virtual void set_is_dynamic(bool is_dynamic) = 0;

 protected:
  TensorDesc() = default;
};

class NaiveTensorDesc final : public TensorDesc {
 public:
  NaiveTensorDesc() = default;
  ~NaiveTensorDesc() override = default;
  NaiveTensorDesc(const NaiveTensorDesc&);
  NaiveTensorDesc(const BlobDescProto&);

  NaiveTensorDesc& operator=(const BlobDescProto&);

  const Shape& shape() const override { return shape_; }
  void set_shape(const Shape& shape) override { shape_ = shape; }
  const Stride& stride() const override { return stride_; }
  void set_stride(const Stride& stride) override { stride_ = stride; }
  DataType data_type() const override { return data_type_; }
  void set_data_type(DataType data_type) override { data_type_ = data_type; }

  bool is_dynamic() const override { return is_dynamic_; }
  void set_is_dynamic(bool is_dynamic) override { is_dynamic_ = is_dynamic; }

 private:
  Shape shape_;
  Stride stride_;
  DataType data_type_;
  bool is_dynamic_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TENSOR_DESC_H_
