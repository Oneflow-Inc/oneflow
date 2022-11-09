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

#include "oneflow/api/cpp/framework/dtype.h"
#include <map>

namespace oneflow_api {

namespace {

std::map<DType, int32_t> DTypeSize = {
    {DType::kFloat, sizeof(float)},   {DType::kDouble, sizeof(double)},
    {DType::kInt8, sizeof(int8_t)},   {DType::kInt32, sizeof(int32_t)},
    {DType::kInt64, sizeof(int64_t)}, {DType::kBool, sizeof(bool)},
};

}

int32_t GetDTypeSize(DType dtype) { return DTypeSize[dtype]; }

}  // namespace oneflow_api
