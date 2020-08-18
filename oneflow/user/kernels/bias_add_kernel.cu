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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/bias_add_kernel.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

namespace {

template<typename T, typename Index>
__global__ void BiasAddGpu(const Index elem_cnt, const Index bias_size, const Index inner_size,
                           const T* x, const T* bias, T* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) { y[i] = x[i] + bias[(i % block_size) / inner_size]; }
}

template<typename Index>
__global__ void BiasAddGpuHalf(const Index elem_cnt, const Index bias_size, const Index inner_size,
                               const half* x, const half* bias, half* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    y[i] = __hadd(x[i], bias[(i % block_size) / inner_size]);
  }
}

template<typename T, typename Index>
__global__ void InplaceBiasAddGpu(const Index elem_cnt, const Index bias_size,
                                  const Index inner_size, const T* bias, T* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) { y[i] += bias[(i % block_size) / inner_size]; }
}

template<typename T, typename Index>
typename std::enable_if<IsFloating<T>::value || std::is_same<T, float16>::value>::type
InplaceBiasAdd(DeviceCtx* ctx, Index outer_size, Index bias_size, Index inner_size, const T* x,
               const T* bias, T* y) {
  CudnnTensorDesc c_desc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, outer_size, bias_size,
                         inner_size, 1);
  CudnnTensorDesc a_desc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, 1, bias_size, 1, 1);
  OF_CUDNN_CHECK(cudnnAddTensor(ctx->cudnn_handle(), CudnnSPOnePtr<float>(), a_desc.Get(), bias,
                                CudnnSPOnePtr<float>(), c_desc.Get(), y));
}

template<typename T, typename Index>
typename std::enable_if<IsIntegral<T>::value>::type InplaceBiasAdd(DeviceCtx* ctx, Index outer_size,
                                                                   Index bias_size,
                                                                   Index inner_size, const T* x,
                                                                   const T* bias, T* y) {
  const Index elem_cnt = outer_size * bias_size * inner_size;
  RUN_CUDA_KERNEL((InplaceBiasAddGpu<T, Index>), ctx, elem_cnt, elem_cnt, bias_size, inner_size,
                  bias, y);
}

}  // namespace

template<typename T, typename Index>
struct BiasAddCalculation<DeviceType::kGPU, T, Index> {
  static void Invoke(DeviceCtx* ctx, Index outer_size, Index bias_size, Index inner_size,
                     const T* x, const T* bias, T* y) {
    const Index elem_cnt = outer_size * bias_size * inner_size;
    if (x == y) {
      InplaceBiasAdd<T, Index>(ctx, outer_size, bias_size, inner_size, x, bias, y);
    } else {
      RUN_CUDA_KERNEL((BiasAddGpu<T, Index>), ctx, elem_cnt, elem_cnt, bias_size, inner_size, x,
                      bias, y);
    }
  }
};

template<typename Index>
struct BiasAddCalculation<DeviceType::kGPU, float16, Index> {
  static void Invoke(DeviceCtx* ctx, Index outer_size, Index bias_size, Index inner_size,
                     const float16* x, const float16* bias, float16* y) {
    if (x == y) {
      InplaceBiasAdd<float16, Index>(ctx, outer_size, bias_size, inner_size, x, bias, y);
    } else {
      const Index elem_cnt = outer_size * bias_size * inner_size;
      RUN_CUDA_KERNEL((BiasAddGpuHalf<Index>), ctx, elem_cnt, elem_cnt, bias_size, inner_size,
                      reinterpret_cast<const half*>(x), reinterpret_cast<const half*>(bias),
                      reinterpret_cast<half*>(y));
    }
  }
};

REGISTER_BIAS_ADD_USER_KERNEL(GPU, float16)
REGISTER_BIAS_ADD_USER_KERNEL(GPU, float)
REGISTER_BIAS_ADD_USER_KERNEL(GPU, double)
REGISTER_BIAS_ADD_USER_KERNEL(GPU, int8_t)
REGISTER_BIAS_ADD_USER_KERNEL(GPU, int32_t)
REGISTER_BIAS_ADD_USER_KERNEL(GPU, int64_t)

}  // namespace oneflow
