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
#ifndef ONEFLOW_API_CPP_FRAMEWORK_DTYPE_H_
#define ONEFLOW_API_CPP_FRAMEWORK_DTYPE_H_

#include <cstdint>

namespace oneflow_api {

enum class DType {
  kInvalidDataType = 0,
  kChar = 1,
  kFloat = 2,
  kDouble = 3,
  kInt8 = 4,
  kInt32 = 5,
  kInt64 = 6,
  kUInt8 = 7,
  kOFRecord = 8,
  kFloat16 = 9,
  kTensorBuffer = 10,
  kBFloat16 = 11,
  kBool = 12,
  kMaxDataType = 13
};

[[nodiscard]] int32_t GetDTypeSize(DType dtype);

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_FRAMEWORK_DTYPE_H_
