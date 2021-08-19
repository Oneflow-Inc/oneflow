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
#ifndef ONEFLOW_API_CPP_TENSOR_TENSOR_H_
#define ONEFLOW_API_CPP_TENSOR_TENSOR_H_

#include <vector>
#include <stdint.h>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow {

class Tensor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Tensor);
  explicit Tensor();
  ~Tensor();

  char* mutable_data() { return this->data_; }
  const char* data() const { return this->data_; }
  const int64_t* shape() const { return this->shape_.data(); }
  int64_t num_axes() const { return this->shape_.size(); }
  int64_t num_elems() const {
    int64_t elems = 1;
    for(auto dim : this->shape_) {
      elems *= dim;
    }
    return elems;
  }
  DataType dtype() const { return this->dtype_; }
  bool is_mutable() { return this->is_mutable_; }
  
  void fromBlob(char* blob_data, 
                const std::vector<int64_t>& shape, 
                DataType dtype,
                bool zero_copy);

 private:
  bool is_mutable_;
  char* data_;
  std::vector<int64_t> shape_;
  DataType dtype_;
};

}  // namespace oneflow

#endif  // ONEFLOW_API_CPP_TENSOR_TENSOR_H_
