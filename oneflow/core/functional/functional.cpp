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

#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {
namespace one {
namespace functional {

Maybe<one::Tensor> Add(const std::shared_ptr<one::Tensor>& x,
                       const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& op = CHECK_JUST(FunctionLibrary::Global()->find("add"));
  return op->call<Maybe<one::Tensor>>(x, y);
}

Maybe<one::Tensor> AddN(const TensorTuple& inputs) {
  static thread_local const auto& op = CHECK_JUST(FunctionLibrary::Global()->find("add_n"));
  return op->call<Maybe<one::Tensor>>(inputs);
}

Maybe<one::Tensor> AddScalar(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) {
  static thread_local const auto& op = CHECK_JUST(FunctionLibrary::Global()->find("add_scalar"));
  return op->call<Maybe<one::Tensor>>(x, scalar);
}

Maybe<one::Tensor> Normalization(const std::shared_ptr<one::Tensor>& x,
                                 const std::shared_ptr<one::Tensor>& moving_mean,
                                 const std::shared_ptr<one::Tensor>& moving_variance,
                                 const std::shared_ptr<one::Tensor>& gamma,
                                 const std::shared_ptr<one::Tensor>& beta, const int32_t& axis,
                                 const float& epsilon, const float& momentum,
                                 const bool& is_training) {
  static thread_local const auto& op = CHECK_JUST(FunctionLibrary::Global()->find("normalization"));
  return op->call<Maybe<one::Tensor>>(x, moving_mean, moving_variance, gamma, beta, axis, epsilon,
                                      momentum, is_training);
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
