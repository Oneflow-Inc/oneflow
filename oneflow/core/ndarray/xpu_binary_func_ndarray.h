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
#ifndef ONEFLOW_CORE_NDARRAY_XPU_BINARY_FUNC_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_BINARY_FUNC_NDARRAY_H_

#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<typename T, template<typename> class binary_func, typename A, typename B>
class XpuBinaryFuncNdarray final {
 public:
  OF_DEVICE_FUNC XpuBinaryFuncNdarray(const A& a, const B& b) : a_(a), b_(b) {}

  template<int NDIMS>
  OF_DEVICE_FUNC typename BinaryFuncTrait<binary_func, T>::return_type Get(int64_t offset) const {
    return binary_func<T>::Invoke(a_.template Get<NDIMS>(offset), b_.template Get<NDIMS>(offset));
  }

 private:
  const A a_;
  const B b_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_BINARY_FUNC_NDARRAY_H_
