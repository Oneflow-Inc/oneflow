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

Tensor::Tensor(Shape shape, DataType dtype)
  : shape_(shape), dtype_(dtype){
  int64_t num_elems = shape.elem_cnt();
  size_t num_bytes = DType::Get(dtype).GetOrThrow()->bytes().GetOrThrow();
  this->data_ = new char[num_items*num_bytes];
}

Tensor::~Tensor() { 
  delete this->data_;
}

Tensor::CopyFrom(const char* data) {
  int64_t num_elems = this->shape_.elem_cnt();
  size_t num_bytes = DType::Get(this->dtype_).GetOrThrow()->bytes().GetOrThrow();
  std::copy(data, data + num_elems * num_bytes, this->data_);
}

static std::shared_ptr<Tensor> Tensor::fromBlob(char* blob_data, 
                                                Shape shape, 
                                                DataType dtype) {
  std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(shape, dtype);
  tensor->CopyFrom(blob_data);
  return tensor;
}

}  // namespace oneflow