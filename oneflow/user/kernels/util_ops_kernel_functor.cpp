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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/user/kernels/util_ops_kernel_functor.h"

namespace oneflow {
namespace user_op {

template<typename T>
struct IsNanFunctor<DeviceType::kCPU, T> {
  void operator()(ep::Stream* stream, bool* y_ptr, const T* x_ptr, const size_t elem_cnt) {
    for (size_t i = 0; i < elem_cnt; i++) { y_ptr[i] = std::isnan(x_ptr[i]); }
  }
};

template<typename T>
struct IsInfFunctor<DeviceType::kCPU, T> {
  void operator()(ep::Stream* stream, bool* y_ptr, const T* x_ptr, const size_t elem_cnt) {
    for (size_t i = 0; i < elem_cnt; i++) { y_ptr[i] = std::isinf(x_ptr[i]); }
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_Util_OPS_FUNCTOR, (DeviceType::kCPU),
                                 UTIL_OPS_FUNCTOR_DTYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
