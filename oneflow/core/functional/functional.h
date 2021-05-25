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

// TODO(): Generate this file automatically.

#ifndef ONEFLOW_CORE_FUNCTIONAL_FUNCTIONAL_H_
#define ONEFLOW_CORE_FUNCTIONAL_FUNCTIONAL_H_

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/scalar.h"

namespace oneflow {
namespace one {
namespace functional {

Maybe<one::Tensor> Add(const std::shared_ptr<one::Tensor>& x,
                       const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> AddN(const TensorTuple& inputs);

Maybe<one::Tensor> AddScalar(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar);

Maybe<one::Tensor> Normalization(const std::shared_ptr<one::Tensor>& x,
                                 const std::shared_ptr<one::Tensor>& moving_mean,
                                 const std::shared_ptr<one::Tensor>& moving_variance,
                                 const std::shared_ptr<one::Tensor>& gamma,
                                 const std::shared_ptr<one::Tensor>& beta, const int32_t& axis,
                                 const float& epsilon, const float& momentum,
                                 const bool& is_training);

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_FUNCTIONAL_H_
