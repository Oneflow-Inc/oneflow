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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/gumbel_softmax_kernel_util.h"

namespace oneflow {
namespace user_op {

namespace {

template<typename T>
__global__ void GumbelSoftmaxAddNoiseForwardGpu(const int64_t n, const double tau, const T* in,
                                                const T* gumbel_noise, T* out) {
  const T minus_one = -1.0;
  const T one = 1.0;
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    const T noise = minus_one * SafeLog(minus_one * SafeLog(one - gumbel_noise[i]));
    out[i] = (in[i] + noise) / static_cast<T>(tau);
  }
}

template<typename T>
__global__ void GumbelSoftmaxAddNoiseForwardGpuHalf(const int64_t n, const double tau, const half* in,
                                                    const half* gumbel_noise, half* out) {
  const half minus_one = __float2half(-1.f);
  const half one = __float2half(1.f);
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    const half noise = minus_one * hlog(minus_one * hlog(one - gumbel_noise[i]));
    out[i] = __hdiv((in[i] + noise), __double2half(tau));
  }
}

}  //  namespace

template<typename T>
struct GumbelSoftmaxAddNoiseImpl<DeviceType::kCUDA, T> {
  static void Forward(ep::Stream* stream, const double tau, int64_t elem_cnt, const T* in_ptr,
                      const T* gumbel_noise_ptr, T* out_ptr) {
    GumbelSoftmaxAddNoiseForwardGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock,
                                      0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
      elem_cnt, tau, in_ptr, gumbel_noise_ptr, out_ptr);
  }
};

template<>
struct GumbelSoftmaxAddNoiseImpl<DeviceType::kCUDA, float16> {
  static void Forward(ep::Stream* stream, const double tau, int64_t elem_cnt, const float16* in_ptr,
                      const float16* gumbel_noise_ptr, float16* out_ptr) {
    GumbelSoftmaxAddNoiseForwardGpuHalf<float16><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                      stream->As<ep::CudaStream>()->cuda_stream()>>>(
        elem_cnt, tau, reinterpret_cast<const half*>(in_ptr),
        reinterpret_cast<const half*>(gumbel_noise_ptr), reinterpret_cast<half*>(out_ptr));
  }
};

#define INITIATE_GUMBEL_SOFTMAX_KERNEL_UTIL_IMPL_CUDA(dtype_pair) \
  template struct GumbelSoftmaxAddNoiseImpl<DeviceType::kCUDA, OF_PP_PAIR_FIRST(dtype_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GUMBEL_SOFTMAX_KERNEL_UTIL_IMPL_CUDA,
                                 GUMBEL_SOFTMAX_KERNEL_DATA_TYPE_SEQ_CUDA);
#undef INITIATE_GUMBEL_SOFTMAX_KERNEL_UTIL_IMPL_CUDA

}  //  namespace user_op
}  //  namespace oneflow
