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
  explicit TensorProcessor(){};
  TensorProcessor& AddInputs(const TensorTuple& init_list);
  TensorProcessor& AddInputs(const TensorTuple& init_list, Symbol<DType> lowest_dtype);

  TensorProcessor& Apply();
  void ComputeCommonDType();
  void CheckHasDifferentInputDType();
  void InsertCast();
  void promote_inputs_to_common_dtype(bool is_promote);
  void InferLowestDType();
  TensorTuple& GetInputs() { return tensor_tuple_; };

 private:
  TensorTuple tensor_tuple_;
  Symbol<DType> common_dtype_ = DType::InvalidDataType();
  std::vector<Symbol<DType>> lowest_dtype_;

  bool has_different_input_dtype_ = false;
  bool promote_inputs_to_common_dtype_ = false;
  bool has_lowest_dtype_ = false;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_TENSOR_PROCESSOR_H_
