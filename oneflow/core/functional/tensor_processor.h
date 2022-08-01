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
#ifndef ONEFLOW_CORE_FUNCTIONAL_TENSOR_PROCESSOR_H_
#define ONEFLOW_CORE_FUNCTIONAL_TENSOR_PROCESSOR_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace oneflow {
namespace one {
namespace functional {

class TensorProcessor final {
 public:
  TensorProcessor()
      : common_dtype_(DType::InvalidDataType()), promote_inputs_to_common_dtype_(false){};
  TensorProcessor& AddInputs(const TensorTuple& init_list);
  TensorProcessor& AddInputs(const TensorTuple& init_list, Symbol<DType> tensor_lowest_dtype);

  Maybe<void> Apply();
  TensorProcessor& PromoteInputsToCommonDtype(bool is_promote);
  Maybe<TensorTuple&> GetInputs() { return tensor_tuple_; };

 private:
  TensorTuple tensor_tuple_;
  Symbol<DType> common_dtype_;
  std::vector<Symbol<DType>> inputs_lowest_dtype_vec_;

  bool promote_inputs_to_common_dtype_;
};

class TensorLayoutProcessor final {
 public:
  TensorLayoutProcessor(const TensorTuple& inputs, bool non_contiguous_enabled)
      : TensorLayoutProcessor(inputs, nullptr, non_contiguous_enabled) {}
  TensorLayoutProcessor(const TensorTuple& inputs, TensorTuple* outputs,
                        bool non_contiguous_enabled)
      : inputs_(inputs),
        outputs_(outputs),
        non_contiguous_enabled_(non_contiguous_enabled),
        converted_(false) {}

  Maybe<void> Apply();

  const TensorTuple& inputs() const {
    if (converted_) { return contiguous_inputs_; }
    return inputs_;
  }
  TensorTuple* outputs() const { return outputs_; }

 private:
  const TensorTuple& inputs_;
  TensorTuple* outputs_;
  bool non_contiguous_enabled_;
  bool converted_;
  TensorTuple contiguous_inputs_;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_TENSOR_PROCESSOR_H_
