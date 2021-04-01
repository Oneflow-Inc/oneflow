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
#include "oneflow/core/framework/tensor_desc.h"

namespace oneflow {

namespace user_op {

TensorDesc& TensorDesc::operator=(const TensorDesc& rhs) {
  *this->mut_shape() = rhs.shape();
  *this->mut_data_type() = rhs.data_type();
  *this->mut_is_dynamic() = rhs.is_dynamic();
  return *this;
}

bool TensorDesc::operator==(const TensorDesc& rhs) const {
  return (this->shape() == rhs.shape()) && (this->data_type() == rhs.data_type())
         && (this->is_dynamic() == rhs.is_dynamic());
}

NaiveTensorDesc::NaiveTensorDesc(const NaiveTensorDesc& rhs) { *this = rhs; }

NaiveTensorDesc::NaiveTensorDesc(const BlobDescProto& proto) { *this = proto; }

NaiveTensorDesc& NaiveTensorDesc::operator=(const BlobDescProto& proto) {
  data_type_ = proto.body().data_type();
  shape_ = Shape(proto.body().shape());
  is_dynamic_ = proto.is_dynamic();
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
