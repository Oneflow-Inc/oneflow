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
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {
namespace functional {

namespace {

Symbol<DType> ComputeCommonDType(const TensorTuple& tensor_tuple) {
  Symbol<DType> common_dtype = DType::InvalidDataType();
  for (auto& tensor_ptr : tensor_tuple) {
    common_dtype = promoteTypes(tensor_ptr->dtype(), common_dtype);
  }
  return common_dtype;
}

bool CheckHasDifferentInputDType(const TensorTuple& tensor_tuple) {
  Symbol<DType> common_dtype = DType::InvalidDataType();
  for (auto& tensor_ptr : tensor_tuple) {
    if (common_dtype == DType::InvalidDataType()) {
      common_dtype = tensor_ptr->dtype();  // Initialize the common_dtype_
    } else {
      return true;
    }
  }
  return false;
}

Maybe<void> InsertCast(TensorTuple& tensor_tuple, const Symbol<DType>& common_dtype) {
  for (auto& tensor_ptr : tensor_tuple) {
    if (tensor_ptr->dtype() != common_dtype) {
      tensor_ptr = JUST(functional::Cast(tensor_ptr, common_dtype));
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

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
  return *this;
}

TensorProcessor& TensorProcessor::PromoteInputsToCommonDtype(bool is_promote) {
  promote_inputs_to_common_dtype_ = is_promote;
  return *this;
}

Maybe<void> TensorProcessor::Apply() {
  if (promote_inputs_to_common_dtype_) {
    bool has_different_input_dtype = CheckHasDifferentInputDType(tensor_tuple_);
    if (has_different_input_dtype) {
      // Cast all the inputs to common_dtype if the input tensor dtype is lower than common_dtype.
      common_dtype_ = ComputeCommonDType(tensor_tuple_);
      InsertCast(tensor_tuple_, common_dtype_);
    }
  } else {
    for (int i = 0; i < tensor_tuple_.size(); ++i) {
      // Cast all the inputs to it's attribute `lowest_dtype` if the input tensor dtype is lower
      // than attribute `lowest_dtype`.
      Symbol<DType> curr_dtype = lowest_dtype_.at(i);
      if (curr_dtype
          && DType::priority_order[curr_dtype->data_type()]
                 > DType::priority_order[tensor_tuple_.at(i)->dtype()->data_type()]) {
        tensor_tuple_.at(i) = JUST(one::functional::Cast(tensor_tuple_.at(i), curr_dtype));
      }
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
