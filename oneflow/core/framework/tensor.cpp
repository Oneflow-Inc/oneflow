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
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace user_op {

Tensor::Tensor(Blob* blob) {
  dptr_ = blob->ForceMutDptr();
  shape_ = blob->shape();
  blob_access_checker_ = blob->blob_access_checker();
  if (blob->ForceMutShapeView()) {
    mut_shape_.reset(new MutShapeView(*blob->ForceMutShapeView()));
  } else {
    mut_shape_.reset();
  }
  data_type_ = blob->data_type();
  mem_case_ = &(blob->mem_case());
}

void Tensor::header_access_check() { this->blob_access_checker_->CheckHeaderMutable(); }

void Tensor::body_access_check() { this->blob_access_checker_->CheckBodyMutable(); }

void Tensor::CopyWithoutData(const Tensor& rhs) {
  dptr_ = rhs.dptr_;
  shape_ = rhs.shape_;
  if (rhs.mut_shape_) {
    mut_shape_.reset(new MutShapeView(*rhs.mut_shape_));
  } else {
    mut_shape_.reset();
  }
  data_type_ = rhs.data_type_;
  mem_case_ = rhs.mem_case_;
  blob_access_checker_ = rhs.blob_access_checker_;
}

Tensor& Tensor::operator=(Tensor&& rhs) {
  dptr_ = rhs.dptr_;
  shape_ = rhs.shape_;
  mut_shape_ = std::move(rhs.mut_shape_);
  data_type_ = rhs.data_type_;
  mem_case_ = rhs.mem_case_;
  blob_access_checker_ = rhs.blob_access_checker_;
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
