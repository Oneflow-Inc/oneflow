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
#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_DESC_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

namespace user_op {

class TensorDesc {
 public:
  virtual ~TensorDesc() = default;

  virtual const Shape& shape() const = 0;
  virtual Shape* mut_shape() = 0;
  virtual DataType data_type() const = 0;
  virtual DataType* mut_data_type() = 0;

  virtual bool is_dynamic() const = 0;
  virtual bool* mut_is_dynamic() = 0;
  virtual void set_is_dynamic(bool val) = 0;

 protected:
  TensorDesc() = default;
};

class NaiveTensorDesc final : public TensorDesc {
 public:
  NaiveTensorDesc() = default;
  ~NaiveTensorDesc() override = default;
  NaiveTensorDesc(const NaiveTensorDesc&);
  NaiveTensorDesc(const BlobDescProto&);

  NaiveTensorDesc& operator=(const NaiveTensorDesc&);
  NaiveTensorDesc& operator=(const BlobDescProto&);

  bool operator==(const NaiveTensorDesc&) const;

  const Shape& shape() const override { return shape_; }
  Shape* mut_shape() override { return &shape_; }
  DataType data_type() const override { return data_type_; }
  DataType* mut_data_type() override { return &data_type_; }

  bool is_dynamic() const override { return is_dynamic_; }
  bool* mut_is_dynamic() override { return &is_dynamic_; }
  void set_is_dynamic(bool val) override { is_dynamic_ = val; }

 private:
  Shape shape_;
  DataType data_type_;
  bool is_dynamic_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_DESC_H_
