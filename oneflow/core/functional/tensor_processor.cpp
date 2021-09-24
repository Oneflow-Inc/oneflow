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

TensorProcessor& TensorProcessor::AddInputs(const TensorTuple& init_list) {
  tensor_tuple_ = init_list;
  return *this;
}

TensorProcessor& TensorProcessor::AddInputs(const TensorTuple& init_list,
                                            Symbol<DType> lowest_dtype) {
  tensor_tuple_ = init_list;
  lowest_dtype_ = lowest_dtype;
  has_lowest_dtype_ = true;
  return *this;
}

void TensorProcessor::ComputeCommonDType() {
  for (auto& tensor_ptr : tensor_tuple_) {
    common_dtype_ = promoteTypes(tensor_ptr->dtype(), common_dtype_);
  }
}

void TensorProcessor::CheckHasDifferentInputDType() {
  for (auto& tensor_ptr : tensor_tuple_) {
    if (common_dtype_ == DType::InvalidDataType()) {
      common_dtype_ = tensor_ptr->dtype();  // Initialize the common_dtype_
    } else {
      has_different_input_dtype_ = true;
      break;  // Just for set the `has_different_input_dtype` flag
    }
  }
}

void TensorProcessor::InsertCast() {
  for (auto& tensor_ptr : tensor_tuple_) {
    if (tensor_ptr->dtype() != common_dtype_) {
      tensor_ptr = CHECK_JUST(functional::Cast(tensor_ptr, common_dtype_));
    }
  }
}

TensorProcessor& TensorProcessor::Apply() {
  if (promote_inputs_to_common_dtype_) { CheckHasDifferentInputDType(); }

  // Initialize common_dtype_ as the lowest_dtype.
  if (has_lowest_dtype_) { common_dtype_ = lowest_dtype_; }

  // Compute the common dtype and Promote.
  if ((has_different_input_dtype_ && promote_inputs_to_common_dtype_) || has_lowest_dtype_) {
    ComputeCommonDType();
    // If current tensor_dtype != promoted common dtype, we insert a Cast function.
    InsertCast();
  }
  // Promote all the inputs to the lowest dtype.

  return *this;
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
