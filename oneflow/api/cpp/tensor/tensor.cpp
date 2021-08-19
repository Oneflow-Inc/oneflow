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

#include "oneflow/api/cpp/tensor/tensor.h"

namespace oneflow {

Tensor::Tensor() {}

Tensor::~Tensor() { 
  if(this->is_mutable_) {
    delete this->data_;
  }
}

void Tensor::fromBlob(char* blob_data, 
                      const std::vector<int64_t>& shape, 
                      DataType dtype,
                      bool zero_copy) {
  this->shape_.assign(std::begin(shape), std::end(shape));
  this->dtype_ = dtype;

  int64_t num_items = 1;
  for(int64_t dim : shape) {
    num_items *= dim;
  }
  size_t num_bytes = DType::Get().GetOrThrow()->bytes().GetOrThrow();

  if(zero_copy) {
    this->data_ = blob_data;
  } else {
    this->data_ = (char*) malloc(num_items*num_bytes);
    std::copy(blob_data, blob_data + num_items*num_bytes, this->data_);
  }
}

}  // namespace oneflow