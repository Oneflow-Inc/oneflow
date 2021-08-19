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
#include "oneflow/core/common/shape.h"

namespace oneflow {

class Tensor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Tensor);
  explicit Tensor(Shape shape, DataType dtype);
  ~Tensor();

  char* mutable_data() { return this->data_; }
  const char* data() const { return this->data_; }
  int64_t num_axes() const { return this->shape_.NumAxes(); }
  int64_t num_elems() const { return this->shape_.elem_cnt();}
  Shape shape() const { return this->shape_; }
  DataType dtype() const { return this->dtype_; }
  
  void CopyFrom(char* blob_data);
  
  static std::shared_ptr<Tensor> fromBlob(char* blob_data, 
                                          Shape shape, 
                                          DataType dtype);

 private:
  char* data_;
  Shape shape_;
  DataType dtype_;
};

}  // namespace oneflow

#endif  // ONEFLOW_API_CPP_TENSOR_TENSOR_H_
