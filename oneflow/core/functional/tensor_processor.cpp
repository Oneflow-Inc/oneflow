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
#include <glog/logging.h>
#include <cstdio>
#include "oneflow/core/functional/tensor_processor.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/throw.h"
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

TensorProcessor& TensorProcessor::PromoteIntegerInputsToFloatDtype(bool is_promote) {
  promote_integer_inputs_to_float_ = is_promote;
  CHECK_OR_THROW(!promote_integer_inputs_to_float_ || promote_inputs_to_common_dtype_)
      << "when set promote_integer_inputs_to_float to 'True', then promote_inputs_to_common_dtype "
         "should be set to 'True' first!";
  return *this;
}

Maybe<void> TensorProcessor::Apply() {
  if (promote_inputs_to_common_dtype_) {
    bool has_different_input_dtype = CheckHasDifferentInputDType(tensor_tuple_);
    if (has_different_input_dtype) {
      common_dtype_ = ComputeCommonDType(tensor_tuple_);
      if (promote_integer_inputs_to_float_ && common_dtype_->is_integer()) {
        // Promotes common dtype to the default float scalar type, if needed.
        // same to pytorch's computeTypes() in torch/csrc/jit/codegen/cuda/type_promotion.cpp
        common_dtype_ = DType::Float();
      }
      JUST(CastToSameType(tensor_tuple_, common_dtype_));
    } else {
      if (tensor_tuple_.size() == 1 && !tensor_tuple_[0]->dtype()->is_floating_point()) {
        Symbol<DType> cast_dtype = inputs_lowest_dtype_vec_[0]->InvalidDataType()
                                       ? DType::Float()
                                       : inputs_lowest_dtype_vec_[0];
        JUST(CastToSameType(tensor_tuple_, cast_dtype));
      }
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
    if (t && !t->is_contiguous()) { return false; }
  }
  return true;
}

Maybe<void> TensorLayoutProcessor::Apply() {
  if (LazyMode::is_enabled()) { return Maybe<void>::Ok(); }
  if (!non_contiguous_enabled_ && !IsAllContiguous(inputs_)) {
    contiguous_inputs_.resize(inputs_.size());
    for (int i = 0; i < inputs_.size(); ++i) { contiguous_inputs_[i] = inputs_[i]->contiguous(); }
    converted_ = true;
  }
  // inplace operation is not allowed if input is non-contiguous and non-contiguous is
  // not supported for this operation
  if (!non_contiguous_enabled_ && outputs_ && !IsAllContiguous(*outputs_)) {
    post_process_outputs_.reserve(outputs_->size());
    post_process_output_indices_.reserve(outputs_->size());
    for (int i = 0; i < outputs_->size(); ++i) {
      if ((*outputs_)[i] && !(*outputs_)[i]->is_contiguous()) {
        post_process_outputs_.emplace_back((*outputs_)[i]);
        post_process_output_indices_.emplace_back(i);
        (*outputs_)[i] = nullptr;
      }
    }
  }
  return Maybe<void>::Ok();
}

TensorLayoutProcessor::~TensorLayoutProcessor() {
  for (int i = 0; i < post_process_output_indices_.size(); ++i) {
    int output_index = post_process_output_indices_[i];
    CHECK_OR_THROW((*outputs_)[output_index])
        << "the output which index is " << i << " should not be nullptr";
    functional::TensorIndex ellipsis_index;
    ellipsis_index.emplace_back(functional::detail::EllipsisIndex());
    CHECK_JUST(functional::TensorSetItem(post_process_outputs_[i], ellipsis_index,
                                         (*outputs_)[output_index]));
    (*outputs_)[output_index] = post_process_outputs_[i];
  }
}

Maybe<void> TensorAutoCastProcessor::Apply() {
  auto autocast_device_type = autocast::get_autocast_device_type();
  auto autocast_dtype = autocast::get_autocast_dtype();
  if (autocast::is_enabled()
      && autocast_meta_.is_autocast_eligible(autocast_device_type, autocast_dtype)) {
    const auto& args_eligible = autocast_meta_.is_args_autocast_eligible();
    CHECK_EQ_OR_RETURN(args_eligible.size(), inputs_.size())
        << Error::RuntimeError() << "argument autocast eligible size should equal to input size";

    auto IsDeviceType = [](const std::shared_ptr<Tensor>& tensor,
                           DeviceType device_type) -> Maybe<bool> {
      return tensor->is_local() ? JUST(tensor->device())->enum_type() == device_type
                                : JUST(tensor->parallel_desc())->device_type() == device_type;
    };
    autocast_inputs_.resize(inputs_.size());
    for (int i = 0; i < inputs_.size(); ++i) {
      if (args_eligible[i] && JUST(IsDeviceType(inputs_[i], autocast_device_type))
          && inputs_[i]->dtype() != autocast_dtype) {
        autocast_inputs_[i] = JUST(functional::To(inputs_[i], autocast_dtype, /*copy*/ false));
      } else {
        autocast_inputs_[i] = inputs_[i];
      }
    }
    converted_ = true;
  }
  return Maybe<void>::Ok();
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
