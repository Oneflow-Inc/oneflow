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

TensorProcessor& TensorProcessor::AddInputs(const TensorTuple& init_tensor_or_tuple) {
  tensor_tuple_.insert(tensor_tuple_.end(), init_tensor_or_tuple.begin(),
                       init_tensor_or_tuple.end());
  return *this;
}

TensorProcessor& TensorProcessor::AddInputs(const TensorTuple& init_tensor_or_tuple,
                                            Symbol<DType> lowest_dtype) {
  tensor_tuple_.insert(tensor_tuple_.end(), init_tensor_or_tuple.begin(),
                       init_tensor_or_tuple.end());
  lowest_dtype_.emplace_back(lowest_dtype);
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

void TensorProcessor::InferLowestDType() {
  for (auto& dtype : lowest_dtype_) {
    if (common_dtype_ == DType::InvalidDataType()) {
      common_dtype_ = dtype;  // Initialize the common_dtype_
    } else {
      common_dtype_ = promoteTypes(dtype, common_dtype_);
    }
  }
}

Maybe<void> TensorProcessor::InsertCast() {
  for (auto& tensor_ptr : tensor_tuple_) {
    if (tensor_ptr->dtype() != common_dtype_) {
      tensor_ptr = JUST(functional::Cast(tensor_ptr, common_dtype_));
    }
  }
  return Maybe<void>::Ok();
}

TensorProcessor& TensorProcessor::PromoteInputsToCommonDtype(bool is_promote) {
  promote_inputs_to_common_dtype_ = is_promote;
  return *this;
}

Maybe<TensorProcessor&> TensorProcessor::Apply() {
  if (promote_inputs_to_common_dtype_) { CheckHasDifferentInputDType(); }

  if (has_lowest_dtype_) {
    InferLowestDType();
  }  // First Infer the "highest" Dtype from each input's lowest DType.

  // Compute the common dtype and Promote.
  if ((has_different_input_dtype_ && promote_inputs_to_common_dtype_) || has_lowest_dtype_) {
    ComputeCommonDType();
    // If current tensor_dtype != promoted common dtype, we insert a Cast function.
    JUST(InsertCast());
  }
  // Promote all the inputs to the lowest dtype.

  return *this;
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
