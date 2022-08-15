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
#ifndef ONEFLOW_CORE_EAGER_CALL_CONTEXT_H_
#define ONEFLOW_CORE_EAGER_CALL_CONTEXT_H_

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/small_vector.h"

namespace oneflow {

namespace one {

class StatefulLocalOpKernel;
class GlobalTensorInferResult;

}  // namespace one

class DeviceCtx;

namespace eager {

class TmpTensor final : public user_op::Tensor {
 public:
  explicit TmpTensor(const std::shared_ptr<MemoryCase>& mem_case)
      : mem_case_(mem_case), tmp_buffer_size_(0), tmp_buffer_ptr_(nullptr) {}
  ~TmpTensor() {}

  ShapeView shape_view() const override { return ShapeView(&tmp_buffer_size_, 1); }
  MutShapeView mut_shape_view() override { return MutShapeView(&tmp_buffer_size_, 1); }
  const Stride& stride() const override {
    UNIMPLEMENTED() << "TmpTensor::stride() is not implemented.";
  }
  DataType data_type() const override { return DataType::kChar; }
  const MemoryCase& mem_case() const override { return *mem_case_; }
  const void* raw_dptr() const override { return tmp_buffer_ptr_; }
  void* mut_raw_dptr() override { return tmp_buffer_ptr_; }

  int64_t tmp_buffer_size() const { return tmp_buffer_size_; }
  void set_tmp_buffer_size(int64_t val) { tmp_buffer_size_ = val; }

  char* mut_tmp_buffer_ptr() { return tmp_buffer_ptr_; }

  void set_tmp_buffer_ptr(char* ptr) { tmp_buffer_ptr_ = ptr; }

 private:
  std::shared_ptr<MemoryCase> mem_case_;
  int64_t tmp_buffer_size_;
  char* tmp_buffer_ptr_;
};

class CallContext {
 public:
  CallContext(ComposedAttrMap&& composed_attrs, vm::EagerBlobObjectList&& inputs,
              vm::EagerBlobObjectList&& outputs,
              const std::shared_ptr<const one::GlobalTensorInferResult>& global_tensor_infer_result,
              const one::OpExprInterpContext& op_interp_ctx,
              const std::shared_ptr<MemoryCase>& mem_case)
      : composed_attrs_(std::move(composed_attrs)),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        global_tensor_infer_result_(global_tensor_infer_result),
        op_interp_ctx_(op_interp_ctx),
        tmp_tensor_(mem_case) {}

  ~CallContext() = default;

  const ComposedAttrMap& composed_attrs() const { return composed_attrs_; }
  const vm::EagerBlobObjectList& inputs() const { return inputs_; }
  const vm::EagerBlobObjectList& outputs() const { return outputs_; }
  const std::shared_ptr<const one::GlobalTensorInferResult>& global_tensor_infer_result() const {
    return global_tensor_infer_result_;
  }
  const one::OpExprInterpContext& op_interp_ctx() const { return op_interp_ctx_; }
  TmpTensor* mut_tmp_tensor() { return &tmp_tensor_; }

 private:
  const ComposedAttrMap composed_attrs_;
  const vm::EagerBlobObjectList inputs_;
  const vm::EagerBlobObjectList outputs_;
  const std::shared_ptr<const one::GlobalTensorInferResult> global_tensor_infer_result_;
  const one::OpExprInterpContext op_interp_ctx_;
  TmpTensor tmp_tensor_;
};

}  // namespace eager

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_CALL_CONTEXT_H_
