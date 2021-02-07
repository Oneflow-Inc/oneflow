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
#ifndef ONEFLOW_CORE_COMMON_FP16_DATA_TYPE_H_
#define ONEFLOW_CORE_COMMON_FP16_DATA_TYPE_H_
#include <type_traits>

// TODO: auto generated
#include "oneflow/core/framework/device_register_gpu.h"
#include "oneflow/core/framework/device_register_cpu.h"

namespace oneflow {
// Type Trait: IsFloat16
template<typename T>
struct IsFloat16 : std::false_type {};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FP16_DATA_TYPE_H_
