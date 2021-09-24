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
#include "oneflow/core/functional/tensor_processor.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {
namespace functional {

TensorProcessor::TensorProcessor(const TensorProcessorConfig& tensor_process_config) {
  config_ = tensor_process_config;
}

TensorProcessor& TensorProcessor::AddInput(const std::shared_ptr<one::Tensor>& tensor_ptr) {
  tensor_ptr_vec.push_back(tensor_ptr);
  return *this;
}

void TensorProcessor::ComputeCommonDType() {
  for (auto& tensor_ptr : tensor_ptr_vec) {
    common_dtype_ = promoteTypes(tensor_ptr->dtype(), common_dtype_);
  }
}

void TensorProcessor::CheckHasDifferentInputDType() {
  for (auto& tensor_ptr : tensor_ptr_vec) {
    if (common_dtype_ == DType::InvalidDataType()) {
      common_dtype_ = tensor_ptr->dtype();  // Initialize the common_dtype_
    } else {
      has_different_input_dtype_ = true;
      break;  // Just for set the `has_different_input_dtype` flag
    }
  }
}

TensorProcessor& TensorProcessor::Apply() {
  if (config_.promote_inputs_to_common_dtype_) { CheckHasDifferentInputDType(); }
  // Compute the common dtype and Promote.
  if (has_different_input_dtype_ && config_.promote_inputs_to_common_dtype_) {
    ComputeCommonDType();
    // If current tensor_dtype != promoted common dtype, we insert a Cast function.
    for (auto& tensor_ptr : tensor_ptr_vec) {
      if (tensor_ptr->dtype() != common_dtype_) {
        tensor_ptr = CHECK_JUST(functional::Cast(tensor_ptr, common_dtype_));
      }
    }
  }
  return *this;
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
