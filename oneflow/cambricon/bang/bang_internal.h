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
#ifndef ONEFLOW_CAMBRICON_BANG_BANG_INTERNAL_H_
#define ONEFLOW_CAMBRICON_BANG_BANG_INTERNAL_H_

#include <stdint.h>

namespace oneflow {

template<int N>
struct AddressList {
  const void* address[N];
  int64_t sizes[N];
};

template<typename T>
__mlu_func__ T bang_static_cast(float scalar) {
  return static_cast<T>(scalar);
}

template<>
__mlu_func__ half bang_static_cast<half>(float scalar) {
  return __float2half_rd(scalar);
}

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_BANG_BANG_INTERNAL_H_
