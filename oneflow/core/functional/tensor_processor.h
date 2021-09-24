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
#ifndef ONEFLOW_CORE_FUNCTIONAL_IMPL_TENSOR_PROCESSOR_H_
#define ONEFLOW_CORE_FUNCTIONAL_IMPL_TENSOR_PROCESSOR_H_

#include "oneflow/core/functional/impl/common.h"

namespace oneflow {
namespace one {
namespace functional {

class TensorProcessorConfig {
 public:
  TensorProcessorConfig() = default;
  explicit TensorProcessorConfig(bool promote_inputs_to_common_dtype)
      : promote_inputs_to_common_dtype_(promote_inputs_to_common_dtype){};
  bool promote_inputs_to_common_dtype_ = false;
};

class TensorProcessor {
 public:
  explicit TensorProcessor(const TensorProcessorConfig&);
  TensorProcessor& AddInput(const std::shared_ptr<one::Tensor>&);
  TensorProcessor& Apply();
  void ComputeCommonDType();
  void CheckHasDifferentInputDType();
  std::vector<std::shared_ptr<one::Tensor>>& Get() { return tensor_ptr_vec; };

 private:
  std::vector<std::shared_ptr<one::Tensor>> tensor_ptr_vec;
  TensorProcessorConfig config_;
  Symbol<DType> common_dtype_ = DType::InvalidDataType();
  bool has_different_input_dtype_ = false;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif
