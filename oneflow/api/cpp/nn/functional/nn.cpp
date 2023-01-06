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
#include "oneflow/api/cpp/nn/functional/nn.h"
#include "oneflow/api/cpp/framework/tensor.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow_api {
namespace nn {

namespace of = oneflow;
namespace functional = of::one::functional;

namespace {
of::Scalar IValueToScalar(const IValue& value) {
#define JUDGE(type) \
  if (value.Is##type()) { return value.To##type(); }
  JUDGE(Bool)
  JUDGE(Int32)
  JUDGE(Int64)
  JUDGE(Float)
  JUDGE(Double)
#undef JUDGE
  return of::Scalar();
}
}  // namespace

#define HANDLE_RESULT(result) Tensor(CHECK_JUST(result))

Tensor concat(const std::vector<Tensor>& tensors, const IValue& dim) {
  oneflow::one::TensorTuple tensor_tuple(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    tensor_tuple.at(i) = tensors.at(i).__internal_tensor();
  }
  return HANDLE_RESULT(functional::Concat(tensor_tuple, dim.ToInt64()));
}

Tensor scalar_add(const Tensor& tensor, const IValue& other) {
  return HANDLE_RESULT(
      functional::ScalarAdd(tensor.__internal_tensor(), IValueToScalar(other), 1, false));
}

Tensor scalar_div(const Tensor& tensor, const IValue& other) {
  return HANDLE_RESULT(functional::ScalarDiv(tensor.__internal_tensor(), IValueToScalar(other)));
}

Tensor clamp(const Tensor& tensor, const IValue& min_value, const IValue& max_value) {
  return HANDLE_RESULT(functional::Clamp(tensor.__internal_tensor(), IValueToScalar(min_value),
                                         IValueToScalar(max_value)));
}

Tensor permute(const Tensor& tensor, const std::vector<int32_t>& permute) {
  return HANDLE_RESULT(functional::Permute(tensor.__internal_tensor(), permute));
}

Tensor identity(const Tensor& tensor) {
  return HANDLE_RESULT(functional::Identity(tensor.__internal_tensor()));
}

}  // namespace nn
}  // namespace oneflow_api
