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
#include "oneflow/core/common/tensor_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"

namespace oneflow {

namespace user_op {

TensorDesc& TensorDesc::operator=(const TensorDesc& rhs) {
  this->set_shape(rhs.shape());
  this->set_stride(rhs.stride());
  this->set_data_type(rhs.data_type());
  this->set_is_dynamic(rhs.is_dynamic());
  return *this;
}

bool TensorDesc::operator==(const TensorDesc& rhs) const {
  return (this->shape() == rhs.shape()) && (this->stride() == rhs.stride())
         && (this->data_type() == rhs.data_type()) && (this->is_dynamic() == rhs.is_dynamic());
}

NaiveTensorDesc::NaiveTensorDesc(const NaiveTensorDesc& rhs) { *this = rhs; }

NaiveTensorDesc::NaiveTensorDesc(const BlobDescProto& proto) { *this = proto; }

NaiveTensorDesc& NaiveTensorDesc::operator=(const BlobDescProto& proto) {
  data_type_ = proto.data_type();
  shape_ = Shape(proto.shape());
  stride_ = Stride(proto.stride());
  is_dynamic_ = proto.is_dynamic();
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
