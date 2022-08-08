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
#include "oneflow/core/job/lazy_mode.h"

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

Maybe<void> CastToSameType(TensorTuple& tensor_tuple, const Symbol<DType>& common_dtype) {
  for (auto& tensor_ptr : tensor_tuple) {
    if (tensor_ptr->dtype() != common_dtype) {
      tensor_ptr = JUST(functional::Cast(tensor_ptr, common_dtype, /*pin_memory=*/false));
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

TensorProcessor& TensorProcessor::AddInputs(const TensorTuple& init_tensor_or_tuple) {
  for (const auto& tensor : init_tensor_or_tuple) {
    tensor_tuple_.emplace_back(tensor);
    inputs_lowest_dtype_vec_.emplace_back(DType::InvalidDataType());
  }
  return *this;
}

TensorProcessor& TensorProcessor::AddInputs(const TensorTuple& init_tensor_or_tuple,
                                            Symbol<DType> tensor_lowest_dtype) {
  for (const auto& tensor : init_tensor_or_tuple) {
    tensor_tuple_.emplace_back(tensor);
    inputs_lowest_dtype_vec_.emplace_back(tensor_lowest_dtype);
  }
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
      common_dtype_ = ComputeCommonDType(tensor_tuple_);
      JUST(CastToSameType(tensor_tuple_, common_dtype_));
    }
  } else {
    for (int i = 0; i < tensor_tuple_.size(); ++i) {
      // Cast all the inputs to it's attribute `lowest_dtype` if the input tensor dtype is lower
      // than attribute `lowest_dtype`.
      Symbol<DType> base_dtype = inputs_lowest_dtype_vec_.at(i);
      if (base_dtype->data_type()
          && DType::priority_order[base_dtype->data_type()]
                 > DType::priority_order[tensor_tuple_.at(i)->dtype()->data_type()]) {
        tensor_tuple_[i] =
            JUST(one::functional::Cast(tensor_tuple_[i], base_dtype, /*pin_memory=*/false));
      }
    }
  }
  return Maybe<void>::Ok();
}

static bool IsAllContiguous(const TensorTuple& tensors) {
  for (const auto& t : tensors) {
    if (!t->is_contiguous()) { return false; }
  }
  return true;
}

Maybe<void> TensorLayoutProcessor::Apply() {
  if (LazyMode::is_enabled()) { return Maybe<void>::Ok(); }
  if (!non_contiguous_enabled_ && !IsAllContiguous(inputs_)) {
    // inplace is not allowed if input is non-contiguous
    if (outputs_) {
      size_t len = std::min(inputs_.size(), outputs_->size());
      for (int i = 0; i < len; ++i) {
        // only requires the inplaced input be contiguous
        CHECK_OR_RETURN((*outputs_)[i] != inputs_[i] || inputs_[i]->is_contiguous())
            << Error::RuntimeError()
            << "inplace operation is not allowed if input is non-contiguous and non-contiguous is "
               "not supported for this operation";
      }
    }
    contiguous_inputs_.resize(inputs_.size());
    for (int i = 0; i < inputs_.size(); ++i) { contiguous_inputs_[i] = inputs_[i]->contiguous(); }
    converted_ = true;
  }
  return Maybe<void>::Ok();
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
