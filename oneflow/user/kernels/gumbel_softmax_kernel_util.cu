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
__global__ void GumbelSoftmaxAddNoiseForwardGpu(const int n, const float tau, const T* in,
                                                const T* gumbel_noise, T* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = (in[i] + gumbel_noise[i]) / static_cast<T>(tau); }
}

template<typename T>
__global__ void GumbelSoftmaxNoiseFromUniformGpu(const int n, const T* gumbel_noise, T* out) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = static_cast<T>(-1.0) * SafeLog(static_cast<T>(-1.0) * SafeLog(static_cast<T>(1.0) - gumbel_noise[i]));
  }
}

template<typename T>
struct GumbelSoftmaxAddNoiseImpl<DeviceType::kCUDA, T> final {
  static void Forward(ep::Stream* stream, double tau, int64_t elem_cnt, const T* in_ptr,
                      T* gumbel_noise_ptr, T* out_ptr);
};

template<typename T>
void GumbelSoftmaxAddNoiseImpl<DeviceType::kCUDA, T>::Forward(ep::Stream* stream, double tau,
                                                              int64_t elem_cnt, const T* in_ptr,
                                                              T* gumbel_noise_ptr,
                                                              T* out_ptr) {
  RUN_CUDA_KERNEL((GumbelSoftmaxNoiseFromUniformGpu<T>), stream, elem_cnt, elem_cnt,
                  gumbel_noise_ptr, gumbel_noise_ptr);

  RUN_CUDA_KERNEL((GumbelSoftmaxAddNoiseForwardGpu<T>), stream, elem_cnt, elem_cnt, tau, in_ptr,
                  gumbel_noise_ptr, out_ptr);
}

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GUMBEL_SOFTMAX_KERNEL_UTIL_IMPL, (DeviceType::kCUDA), GUMBEL_SOFTMAX_KERNEL_DATA_TYPE_SEQ);

}  //  namespace oneflow