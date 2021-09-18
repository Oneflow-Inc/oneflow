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
#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

#define MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY(func_name, T) cub::DeviceReduce::func_name<T*, T*>
DEFINE_STATIC_SWITCH_FUNC(cudaError_t, Sum, MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

#undef MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY

#define KU_IF_METHOD                     \
  template<typename T, typename Derived> \
  void GpuKernelUtilIf<T, Derived>::

KU_IF_METHOD InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                uint32_t random_seed, Blob* blob) {
  WithHostBlobAndStreamSynchronizeEnv(ctx, blob, [&](Blob* host_blob) {
    KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, initializer_conf, random_seed,
                                                        host_blob);
  });
}

#define KU_FLOATING_METHOD \
  template<typename T>     \
  void KernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsFloating<T>::value>::type>::

#define KU_INTEGRAL_METHOD \
  template<typename T>     \
  void KernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsIntegral<T>::value>::type>::

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto)                                \
  template struct GpuKernelUtilIf<type_cpp, KernelUtil<DeviceType::kGPU, type_cpp>>; \
  template struct KernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

template<typename T, typename U>
__global__ void CastOnGpu(const T* in, U* out, int64_t elem_num) {
  CUDA_1D_KERNEL_LOOP(i, elem_num) { out[i] = static_cast<U>(in[i]); }
}

template<>
__global__ void CastOnGpu<float, half>(const float* in, half* out, int64_t elem_num) {
  const int64_t elem_num_2 = elem_num / 2;
  const auto* in_2 = reinterpret_cast<const float2*>(in);
  auto* out_2 = reinterpret_cast<half2*>(out);
  CUDA_1D_KERNEL_LOOP(i, elem_num_2) { out_2[i] = __float22half2_rn(in_2[i]); }
  if (elem_num % 2 == 1 && blockIdx.x == 0 && threadIdx.x == 0) {
    out[elem_num - 1] = __float2half(in[elem_num - 1]);
  }
}

template<>
__global__ void CastOnGpu<half, float>(const half* in, float* out, int64_t elem_num) {
  const int64_t elem_num_2 = elem_num / 2;
  const auto* in_2 = reinterpret_cast<const half2*>(in);
  auto* out_2 = reinterpret_cast<float2*>(out);
  CUDA_1D_KERNEL_LOOP(i, elem_num_2) { out_2[i] = __half22float2(in_2[i]); }
  if (elem_num % 2 == 1 && blockIdx.x == 0 && threadIdx.x == 0) {
    out[elem_num - 1] = __half2float(in[elem_num - 1]);
  }
}

template<typename T, typename U>
void CopyElemOnGpu(DeviceCtx* ctx, const T* in_dptr, U* out_dptr, int64_t elem_num) {
  if (elem_num == 0) { return; }
  if (std::is_same<T, U>::value) {
    Memcpy<DeviceType::kGPU>(ctx, out_dptr, in_dptr, elem_num * sizeof(T));
  } else {
    CastOnGpu<T, U>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            in_dptr, out_dptr, elem_num);
  }
}

template<>
void CopyElemOnGpu<float, float16>(DeviceCtx* ctx, const float* in_dptr, float16* out_dptr,
                                   int64_t elem_num) {
  if (RoundUp(elem_num, 2) == 0) { return; }
  CastOnGpu<float, half>
      <<<BlocksNum4ThreadsNum(RoundUp(elem_num, 2) / 2), kCudaThreadsNumPerBlock, 0,
         ctx->cuda_stream()>>>(in_dptr, reinterpret_cast<half*>(out_dptr), elem_num);
}

template<>
void CopyElemOnGpu<float16, float>(DeviceCtx* ctx, const float16* in_dptr, float* out_dptr,
                                   int64_t elem_num) {
  if (RoundUp(elem_num, 2) == 0) { return; }
  CastOnGpu<half, float>
      <<<BlocksNum4ThreadsNum(RoundUp(elem_num, 2) / 2), kCudaThreadsNumPerBlock, 0,
         ctx->cuda_stream()>>>(reinterpret_cast<const half*>(in_dptr), out_dptr, elem_num);
}

#define INSTANTIATE_COPY_ELEM_ON_GPU(T, U) \
  template void CopyElemOnGpu(DeviceCtx* ctx, const T* in_dptr, U* out_dptr, int64_t elem_num);

#define MAKE_COPY_ELEM_ON_GPU_ENTRY(TPair, UPair) \
  INSTANTIATE_COPY_ELEM_ON_GPU(OF_PP_PAIR_FIRST(TPair), OF_PP_PAIR_FIRST(UPair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_COPY_ELEM_ON_GPU_ENTRY, POD_DATA_TYPE_SEQ, POD_DATA_TYPE_SEQ)

}  // namespace oneflow
