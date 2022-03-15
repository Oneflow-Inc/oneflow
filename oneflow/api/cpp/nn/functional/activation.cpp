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
#include "oneflow/api/cpp/nn/functional/activation.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow_api {
namespace nn {

namespace of = oneflow;
namespace functional = of::one::functional;

Tensor relu(const Tensor& tensor) {
  return Tensor(functional::Relu(tensor.__internal_tensor(), false).GetPtrOrThrow());
}

}  // namespace nn
}  // namespace oneflow_api
