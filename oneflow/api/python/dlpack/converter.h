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
#include "oneflow/api/python/dlpack/dlpack.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace one {
class Tensor;
}

Maybe<one::Tensor> fromDLPack(const DLManagedTensor* src);
Maybe<DLManagedTensor*> toDLPack(const std::shared_ptr<one::Tensor>& src);

}  // namespace oneflow
