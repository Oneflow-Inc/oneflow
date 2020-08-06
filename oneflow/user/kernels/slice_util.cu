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
__global__ void SliceForwardGpu(const int n, SliceParams params, SliceIndexHelper entire_idx_cvtr,
                                SliceIndexHelper sliced_idx_cvtr, const T* entire, T* sliced) {
  int64_t nd_index[kSliceMaxDims];
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t offset =
        SliceOffsetToEntireOffset(i, nd_index, params, entire_idx_cvtr, sliced_idx_cvtr);
    sliced[i] = entire[offset];
  }
}

template<typename T>
__global__ void SliceBackwardGpu(const int n, SliceParams params, SliceIndexHelper entire_idx_cvtr,
                                 SliceIndexHelper sliced_idx_cvtr, T* entire, const T* sliced) {
  int64_t nd_index[kSliceMaxDims];
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t offset =
        SliceOffsetToEntireOffset(i, nd_index, params, entire_idx_cvtr, sliced_idx_cvtr);
    entire[offset] = sliced[i];
  }
}

}  // namespace

template<typename T>
struct SliceKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const SliceParams& params, const T* entired, T* sliced) {
    int64_t elem_cnt = 1;
    FOR_RANGE(int, i, 0, params.ndim) { elem_cnt *= params.size[i]; }
    SliceIndexHelper entire_idx_cvtr(params.dims, params.ndim);
    SliceIndexHelper sliced_idx_cvtr(params.size, params.ndim);
    SliceForwardGpu<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, params, entire_idx_cvtr, sliced_idx_cvtr, entired, sliced);
  }

  static void Backward(DeviceCtx* ctx, const SliceParams& params, const T* sliced, T* entired) {
    int64_t elem_cnt = 1;
    FOR_RANGE(int, i, 0, params.ndim) { elem_cnt *= params.size[i]; }
    SliceIndexHelper entire_idx_cvtr(params.dims, params.ndim);
    SliceIndexHelper sliced_idx_cvtr(params.size, params.ndim);
    SliceBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, params, entire_idx_cvtr, sliced_idx_cvtr, entired, sliced);
  }
};

INSTANTIATE_SLICE_KERNEL_UTIL_WITH_DEVICE(DeviceType::kGPU)

}  // namespace oneflow
