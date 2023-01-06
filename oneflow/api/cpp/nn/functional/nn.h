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
#ifndef ONEFLOW_API_CPP_NN_FUNCTIONAL_NN_H_
#define ONEFLOW_API_CPP_NN_FUNCTIONAL_NN_H_
#include "../../framework.h"

namespace oneflow_api {
namespace nn {

Tensor concat(const std::vector<Tensor>& tensors, const IValue& dim);

#define DECLARE_SCALAR_OP(suffix) Tensor scalar_##suffix(const Tensor& tensor, const IValue& other);
DECLARE_SCALAR_OP(add)
DECLARE_SCALAR_OP(div)
#undef DECLARE_SCALAR_OP

Tensor clamp(const Tensor& tensor, const IValue& min_value, const IValue& max_value);

Tensor permute(const Tensor& tensor, const std::vector<int32_t>& permute);

Tensor identity(const Tensor& tensor);

}  // namespace nn

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_NN_FUNCTIONAL_NN_H_
