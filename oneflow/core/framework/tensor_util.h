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
#include "oneflow/core/framework/tensor.h"
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
  if constexpr (GetDataType<T>() == kInt64) {
    if (scalar_tensor->dtype()->data_type() == DataType::kInt8
        || scalar_tensor->dtype()->data_type() == kUInt8) {
      int8_t int8_integer = 0;
      JUST(GetItemInScalarTensor(scalar_tensor, &int8_integer, sizeof(int8_t)));
      scalar = static_cast<T>(int8_integer);
    } else if (scalar_tensor->dtype()->data_type() == DataType::kInt16
               || scalar_tensor->dtype()->data_type() == kUInt16) {
      int16_t int16_integer = 0;
      JUST(GetItemInScalarTensor(scalar_tensor, &int16_integer, sizeof(int16_t)));
      scalar = static_cast<T>(int16_integer);
    } else if (scalar_tensor->dtype()->data_type() == DataType::kInt32
               || scalar_tensor->dtype()->data_type() == kUInt32) {
      int32_t int32_integer = 0;
      JUST(GetItemInScalarTensor(scalar_tensor, &int32_integer, sizeof(int32_t)));
      scalar = static_cast<T>(int32_integer);
    } else if (scalar_tensor->dtype()->data_type() == DataType::kInt64
               || scalar_tensor->dtype()->data_type() == kUInt64) {
      int64_t int64_integer = 0;
      JUST(GetItemInScalarTensor(scalar_tensor, &int64_integer, sizeof(int64_t)));
      scalar = static_cast<T>(int64_integer);
    }
  } else {
    JUST(GetItemInScalarTensor(scalar_tensor, &scalar, sizeof(T)));
  }
  return scalar;
}

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
