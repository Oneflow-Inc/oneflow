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
#include <memory>
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/user_op_tensor.h"


namespace oneflow {
namespace okl {

class TmpBufferMapper final : public oneflow::user_op::Tensor {
 public:
  TmpBufferMapper(user_op::Tensor* tensor, size_t offset) : tensor_(tensor), offset_(offset) {}

  ShapeView shape_view() const override { return tensor_->shape_view(); }
  MutShapeView mut_shape_view() override { return tensor_->mut_shape_view(); }
  const Stride& stride() const override { return tensor_->stride(); }
  DataType data_type() const override { return tensor_->data_type(); }
  const MemoryCase& mem_case() const override { return tensor_->mem_case(); }

  const void* raw_dptr() const override {
    return (reinterpret_cast<const char*>(tensor_->raw_dptr()) + offset_);
  }
  void* mut_raw_dptr() override {
    return (reinterpret_cast<char*>(tensor_->mut_raw_dptr()) + offset_);
  }

 private:
  user_op::Tensor* tensor_;
  size_t offset_;
};

using namespace user_op;
class TmpBufferManager {
 public:
  static std::unordered_map<std::string, size_t> offset_list_;
  static std::shared_ptr<TmpBufferManager> InitTmpBufferManager(user_op::Tensor* tensor);
  static size_t InferTmpSize(user_op::InferContext* ctx);

  explicit TmpBufferManager(user_op::Tensor* tensor);

  user_op::Tensor* FetchTmpBuffer(std::string& op_name) ;

 private:
  std::unordered_map<std::string, TmpBufferMapper> manager_;
};

}  // namespace okl
}  // namespace oneflow

#endif // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_TMP_BUFFER_MANAGER_H_