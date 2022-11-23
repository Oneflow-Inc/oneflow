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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_TMP_BUFFER_MANAGER_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_TMP_BUFFER_MANAGER_H_
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/user_op_tensor.h"

namespace oneflow {
namespace okl {

class TmpBufferManager final : public oneflow::user_op::Tensor {
 public:
  explicit TmpBufferManager(user_op::Tensor* tensor) : tensor_(tensor) {}

  ShapeView shape_view() const override { return tensor_->shape_view(); }
  MutShapeView mut_shape_view() override { return tensor_->mut_shape_view(); }
  const Stride& stride() const override { return tensor_->stride(); }
  DataType data_type() const override { return tensor_->data_type(); }
  const MemoryCase& mem_case() const override { return tensor_->mem_case(); }

  const void* raw_dptr() const override {
    return (reinterpret_cast<const char*>(tensor_->raw_dptr()));
  }
  void* mut_raw_dptr() override { return (reinterpret_cast<char*>(tensor_->mut_raw_dptr())); }

  static std::shared_ptr<TmpBufferManager> InitTmpBufferManager(user_op::Tensor* tensor);
  static size_t InferTmpSize(user_op::InferContext* ctx);

 private:
  user_op::Tensor* tensor_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_TMP_BUFFER_MANAGER_H_