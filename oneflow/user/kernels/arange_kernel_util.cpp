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
#include "oneflow/user/kernels/arange_kernel_util.h"

namespace oneflow {

namespace user_op {
template<typename T>
struct ArangeFunctor<DeviceType::kCPU, T> final {
  void operator()(ep::Stream* stream, const T start, const T delta, const int64_t arange_elem_cnt,
                  T* out) {
    DoArange<T>(start, delta, arange_elem_cnt, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARANGE_FUNCTOR, (DeviceType::kCPU),
                                 ARANGE_DATA_TYPE_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARANGE_FUNCTOR, (DeviceType::kCPU),
                                 FLOAT16_DATA_TYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
