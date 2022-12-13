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
#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_

#include <functional>
#include <string>

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace ep {
class Stream;
}

namespace vm {
class EagerBlobObject;
}

namespace one {

class Tensor;

Maybe<void> SyncAccessTensorWithTimeOut(
    const std::shared_ptr<Tensor>& tensor,
    const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& callback,
    const std::string& modifier);

Maybe<void> CopyLocalTensorDataTo(const std::shared_ptr<Tensor>& input, void* mem_ptr, size_t size);

Maybe<Scope> GetTensorScope(const std::shared_ptr<Tensor>& tensor);

Maybe<void> GetItemInScalarTensor(const std::shared_ptr<Tensor>& scalar_tensor, void* scalar_ptr,
                                  size_t size);
template<typename T>
Maybe<T> GetItemInScalarTensor(const std::shared_ptr<Tensor>& scalar_tensor) {
  T scalar{0};
  JUST(GetItemInScalarTensor(scalar_tensor, &scalar, sizeof(T)));
  return scalar;
}

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
