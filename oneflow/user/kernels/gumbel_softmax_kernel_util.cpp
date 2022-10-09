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
#include "oneflow/user/kernels/gumbel_softmax_kernel_util.h"

namespace oneflow {

template<typename T>
struct GumbelSoftmaxAddNoiseImpl<DeviceType::kCPU, T> final {
  static void Forward(ep::Stream* stream, double tau, int64_t elem_cnt, const T* in_ptr,
                      T* gumbel_noise_ptr , T* out_ptr);
};

template<typename T>
void GumbelSoftmaxAddNoiseImpl<DeviceType::kCPU, T>::Forward(ep::Stream* stream, double tau,
                                                             int64_t elem_cnt, const T* in_ptr,
                                                             T* gumbel_noise_ptr, T* out_ptr) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    gumbel_noise_ptr[i] = static_cast<T>(-1.0) * SafeLog(static_cast<T>(-1.0) * SafeLog(static_cast<T>(1.0) - gumbel_noise_ptr[i]));
  }

  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    out_ptr[i] = (in_ptr[i] + gumbel_noise_ptr[i]) / static_cast<T>(tau);
  }
}

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GUMBEL_SOFTMAX_KERNEL_UTIL_IMPL, (DeviceType::kCPU),
                                 GUMBEL_SOFTMAX_KERNEL_DATA_TYPE_SEQ);

}  //  namespace