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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_TMPBUFFERMANAGER_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_TMPBUFFERMANAGER_H_
#include <unordered_map>
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/user_op_tensor.h"

namespace oneflow {
namespace okl {

class TmpBufferManager {
  class TmpBufferTensor final : public oneflow::user_op::Tensor {
   public:
    explicit TmpBufferTensor(user_op::Tensor* tensor,const user_op::TensorDesc* tensor_desc, int offset)
        : tensor_(tensor),
          raw_dptr_(reinterpret_cast<char*>(tensor_->mut_raw_dptr()) + offset),
          tensor_desc_(tensor_desc) {}

    ShapeView shape_view() const override { return tensor_desc_->shape(); }
    const Stride& stride() const override { return tensor_desc_->stride(); }
    DataType data_type() const override { return tensor_desc_->data_type(); }
    MutShapeView mut_shape_view() override { return tensor_->mut_shape_view(); }
    const MemoryCase& mem_case() const override { return tensor_->mem_case(); }

    const void* raw_dptr() const override { return raw_dptr_; }
    void* mut_raw_dptr() override { return raw_dptr_; }

   private:
    user_op::Tensor* tensor_;
    void* raw_dptr_;
    const user_op::TensorDesc* tensor_desc_;
  };

 public:
  static size_t InferTmpSize(user_op::InferContext* ctx);

  explicit TmpBufferManager(user_op::Tensor* tensor) : tensor_(tensor) {}
  user_op::Tensor* GetBufferTensor(const user_op::TensorDesc* tensor_desc, int offset = 0) {
    auto res = map_.insert({tensor_desc, TmpBufferTensor(tensor_, tensor_desc, offset)}).first;
    return &res->second;
  }

 private:
  std::unordered_map<const user_op::TensorDesc*, TmpBufferTensor> map_{};
  user_op::Tensor* tensor_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_TMPBUFFERMANAGER_H_
