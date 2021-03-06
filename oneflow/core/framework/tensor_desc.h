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

class TensorDesc final {
 public:
  TensorDesc() = default;
  ~TensorDesc() = default;
  TensorDesc(const TensorDesc&);
  TensorDesc(const BlobDescProto&);

  TensorDesc& operator=(const TensorDesc&);
  TensorDesc& operator=(const BlobDescProto&);

  bool operator==(const TensorDesc&) const;

  const Shape& shape() const { return shape_; }
  Shape* mut_shape() { return &shape_; }
  DataType data_type() const { return data_type_; }
  DataType* mut_data_type() { return &data_type_; }

  bool is_dynamic() const { return is_dynamic_; }
  bool* mut_is_dynamic() { return &is_dynamic_; }
  void set_is_dynamic(bool val) { is_dynamic_ = val; }

 private:
  Shape shape_;
  DataType data_type_;
  bool is_dynamic_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_DESC_H_
