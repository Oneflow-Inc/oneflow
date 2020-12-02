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
#ifndef ONEFLOW_CORE_KERNEL_BLOB_TENSOR_VIEW_H_
#define ONEFLOW_CORE_KERNEL_BLOB_TENSOR_VIEW_H_

#include "oneflow/core/framework/tensor.h"

namespace oneflow {

class Blob;

namespace user_op {

class BlobTensorView final : public Tensor {
 public:
  explicit BlobTensorView(Blob* blob);
  ~BlobTensorView() = default;

  const ShapeView& shape() const override;
  MutShapeView* mut_shape() override;
  DataType data_type() const override;
  const MemoryCase& mem_case() const override;
  const void* raw_dptr() const override;
  void* mut_raw_dptr() override;

  void Reset(Blob* blob);

 private:
  Blob* blob_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BLOB_TENSOR_VIEW_H_
