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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_TMP_BUFFER_MANAGER_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_TMP_BUFFER_MANAGER_H_
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include <unordered_map>

namespace oneflow {
namespace okl {

class TmpBufferManager {
  class PoolToTensor final : public oneflow::user_op::Tensor {
   public:
    explicit PoolToTensor(user_op::Tensor* tensor, const user_op::TensorDesc* tensor_desc,
                          int64_t offset)
        : tensor_(tensor),
          raw_dptr_(reinterpret_cast<char*>(tensor_->mut_raw_dptr()) + offset),
          tensor_desc_(tensor_desc) {}

    ShapeView shape_view() const override { return tensor_desc_->shape(); }
    const Stride& stride() const override { return tensor_desc_->stride(); }
    DataType data_type() const override { return tensor_desc_->data_type(); }
    MutShapeView mut_shape_view() override { TODO(); }
    const MemoryCase& mem_case() const override { return tensor_->mem_case(); }

    const void* raw_dptr() const override { return raw_dptr_; }
    void* mut_raw_dptr() override { return raw_dptr_; }

   private:
    user_op::Tensor* tensor_;
    void* raw_dptr_;
    const user_op::TensorDesc* tensor_desc_;
  };

  class PoolToBuffer final : public oneflow::user_op::Tensor {
   public:
    explicit PoolToBuffer(user_op::Tensor* tensor, int64_t size, int64_t offset)
        : tensor_(tensor),
          raw_dptr_(reinterpret_cast<char*>(tensor_->mut_raw_dptr()) + offset),
          shape_({size}) {}

    ShapeView shape_view() const override { return shape_; }
    const Stride& stride() const override { return tensor_->stride(); }
    DataType data_type() const override { return tensor_->data_type(); }
    MutShapeView mut_shape_view() override { return shape_; }
    const MemoryCase& mem_case() const override { return tensor_->mem_case(); }

    const void* raw_dptr() const override { return raw_dptr_; }
    void* mut_raw_dptr() override { return raw_dptr_; }

   private:
    user_op::Tensor* tensor_;
    void* raw_dptr_;
    Shape shape_;
  };

 public:
  static size_t InferTmpSize(user_op::InferContext* ctx);

  explicit TmpBufferManager(user_op::Tensor* tensor) : tensor_(tensor) {}
  user_op::Tensor* GetPoolTensor(const user_op::TensorDesc* tensor_desc, int64_t offset) {
    CHECK_LE(offset + tensor_desc->shape().elem_cnt() * GetSizeOfDataType(tensor_desc->data_type()),
             tensor_->shape_view().elem_cnt());
    auto res = tensor_map_.insert({tensor_desc, PoolToTensor(tensor_, tensor_desc, offset)}).first;
    return &res->second;
  }

  user_op::Tensor* GetPoolBuffer(int64_t size, int64_t offset) {
    auto res = buffer_map_.insert({{size, offset}, PoolToBuffer(tensor_, size, offset)}).first;
    return &res->second;
  }

 private:
  std::unordered_map<const user_op::TensorDesc*, PoolToTensor> tensor_map_{};
  std::unordered_map<std::pair<int64_t, int64_t>, PoolToBuffer> buffer_map_{};
  user_op::Tensor* tensor_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_TMP_BUFFER_MANAGER_H_
