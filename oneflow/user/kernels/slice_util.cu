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
#include "oneflow/user/kernels/slice_util.h"

namespace oneflow {

namespace {

template<typename T>
__device__ __forceinline__ void SliceForward(const int n, const SliceParams& params,
                                             const SliceIndexHelper& entire_idx_cvtr,
                                             const SliceIndexHelper& sliced_idx_cvtr,
                                             const T* entire, T* sliced) {
  int64_t nd_index[kSliceMaxDims];
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t offset =
        SliceOffsetToEntireOffset(i, nd_index, params, entire_idx_cvtr, sliced_idx_cvtr);
    sliced[i] = entire[offset];
  }
}

template<typename T>
__device__ __forceinline__ void SliceBackward(const int n, const SliceParams& params,
                                              const SliceIndexHelper& entire_idx_cvtr,
                                              const SliceIndexHelper& sliced_idx_cvtr, T* entire,
                                              const T* sliced) {
  int64_t nd_index[kSliceMaxDims];
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t offset =
        SliceOffsetToEntireOffset(i, nd_index, params, entire_idx_cvtr, sliced_idx_cvtr);
    entire[offset] = sliced[i];
  }
}

template<typename T>
__global__ void SliceForwardGpu(const int n, SliceParams params, SliceIndexHelper entire_idx_cvtr,
                                SliceIndexHelper sliced_idx_cvtr, const T* entire, T* sliced) {
  SliceForward(n, params, entire_idx_cvtr, sliced_idx_cvtr, entire, sliced);
}

template<typename T>
__global__ void SliceBackwardGpu(const int n, SliceParams params, SliceIndexHelper entire_idx_cvtr,
                                 SliceIndexHelper sliced_idx_cvtr, T* entire, const T* sliced) {
  SliceBackward(n, params, entire_idx_cvtr, sliced_idx_cvtr, entire, sliced);
}

__global__ void SliceForwardGpuHalf(const int n, SliceParams params,
                                    SliceIndexHelper entire_idx_cvtr,
                                    SliceIndexHelper sliced_idx_cvtr, const half* entire,
                                    half* sliced) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif
  SliceForward(n, params, entire_idx_cvtr, sliced_idx_cvtr, entire, sliced);
}

__global__ void SliceBackwardGpuHalf(const int n, SliceParams params,
                                     SliceIndexHelper entire_idx_cvtr,
                                     SliceIndexHelper sliced_idx_cvtr, half* entire,
                                     const half* sliced) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif
  SliceBackward(n, params, entire_idx_cvtr, sliced_idx_cvtr, entire, sliced);
}

}  // namespace

template<typename T>
struct SliceKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const SliceParams& params, const T* entire, T* sliced) {
    int64_t elem_cnt = 1;
    FOR_RANGE(int, i, 0, params.ndim) { elem_cnt *= params.size[i]; }
    SliceIndexHelper entire_idx_cvtr(params.dims, params.ndim);
    SliceIndexHelper sliced_idx_cvtr(params.size, params.ndim);
    SliceForwardGpu<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, params, entire_idx_cvtr, sliced_idx_cvtr, entire, sliced);
  }

  static void Backward(DeviceCtx* ctx, const SliceParams& params, const T* sliced, T* entire) {
    int64_t elem_cnt = 1;
    FOR_RANGE(int, i, 0, params.ndim) { elem_cnt *= params.size[i]; }
    SliceIndexHelper entire_idx_cvtr(params.dims, params.ndim);
    SliceIndexHelper sliced_idx_cvtr(params.size, params.ndim);
    SliceBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, params, entire_idx_cvtr, sliced_idx_cvtr, entire, sliced);
  }
};

template<>
void SliceKernelUtil<DeviceType::kGPU, float16>::Forward(DeviceCtx* ctx, const SliceParams& params,
                                                         const float16* entire, float16* sliced) {
  int64_t elem_cnt = 1;
  FOR_RANGE(int, i, 0, params.ndim) { elem_cnt *= params.size[i]; }
  SliceIndexHelper entire_idx_cvtr(params.dims, params.ndim);
  SliceIndexHelper sliced_idx_cvtr(params.size, params.ndim);
  SliceForwardGpuHalf<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                        ctx->cuda_stream()>>>(elem_cnt, params, entire_idx_cvtr, sliced_idx_cvtr,
                                              reinterpret_cast<const half*>(entire),
                                              reinterpret_cast<half*>(sliced));
}

template<>
void SliceKernelUtil<DeviceType::kGPU, float16>::Backward(DeviceCtx* ctx, const SliceParams& params,
                                                          const float16* sliced, float16* entire) {
  int64_t elem_cnt = 1;
  FOR_RANGE(int, i, 0, params.ndim) { elem_cnt *= params.size[i]; }
  SliceIndexHelper entire_idx_cvtr(params.dims, params.ndim);
  SliceIndexHelper sliced_idx_cvtr(params.size, params.ndim);
  SliceBackwardGpuHalf<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx->cuda_stream()>>>(elem_cnt, params, entire_idx_cvtr, sliced_idx_cvtr,
                                               reinterpret_cast<half*>(entire),
                                               reinterpret_cast<const half*>(sliced));
}

INSTANTIATE_SLICE_KERNEL_UTIL_WITH_DEVICE(DeviceType::kGPU)
INSTANTIATE_SLICE_KERNEL_UTIL(DeviceType::kGPU, float16)

}  // namespace oneflow
