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
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace {

template<typename T, int NDIM>
__global__ void SliceForwardGpu(const int n, SliceParams params,
                                SliceIndexHelper<NDIM> entire_idx_cvtr,
                                SliceIndexHelper<NDIM> sliced_idx_cvtr, const T* entire,
                                T* sliced) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t offset = SliceOffsetToEntireOffset<NDIM>(i, params, entire_idx_cvtr, sliced_idx_cvtr);
    sliced[i] = entire[offset];
  }
}

template<typename T, int NDIM>
__global__ void SliceBackwardGpu(const int n, SliceParams params,
                                 SliceIndexHelper<NDIM> entire_idx_cvtr,
                                 SliceIndexHelper<NDIM> sliced_idx_cvtr, T* entire,
                                 const T* sliced) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t offset = SliceOffsetToEntireOffset<NDIM>(i, params, entire_idx_cvtr, sliced_idx_cvtr);
    entire[offset] = sliced[i];
  }
}

template<int NDIM>
__global__ void SliceForwardGpuHalf(const int n, SliceParams params,
                                    SliceIndexHelper<NDIM> entire_idx_cvtr,
                                    SliceIndexHelper<NDIM> sliced_idx_cvtr, const half* entire,
                                    half* sliced) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t offset = SliceOffsetToEntireOffset<NDIM>(i, params, entire_idx_cvtr, sliced_idx_cvtr);
    sliced[i] = entire[offset];
  }
}

template<int NDIM>
__global__ void SliceBackwardGpuHalf(const int n, SliceParams params,
                                     SliceIndexHelper<NDIM> entire_idx_cvtr,
                                     SliceIndexHelper<NDIM> sliced_idx_cvtr, half* entire,
                                     const half* sliced) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t offset = SliceOffsetToEntireOffset<NDIM>(i, params, entire_idx_cvtr, sliced_idx_cvtr);
    entire[offset] = sliced[i];
  }
}

template<typename T, int NDIM>
void LaunchSliceForward(DeviceCtx* ctx, const SliceParams& params, const T* entire, T* sliced) {
  CHECK_EQ(params.ndim, NDIM);
  int64_t elem_cnt = params.elem_cnt();
  SliceIndexHelper<NDIM> entire_idx_cvtr(params.dims);
  SliceIndexHelper<NDIM> sliced_idx_cvtr(params.size);
  SliceForwardGpu<T, NDIM>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, params, entire_idx_cvtr, sliced_idx_cvtr, entire, sliced);
}

template<typename T, int NDIM>
void LaunchSliceBackward(DeviceCtx* ctx, const SliceParams& params, const T* sliced, T* entire) {
  CHECK_EQ(params.ndim, NDIM);
  int64_t elem_cnt = params.elem_cnt();
  SliceIndexHelper<NDIM> entire_idx_cvtr(params.dims);
  SliceIndexHelper<NDIM> sliced_idx_cvtr(params.size);
  SliceBackwardGpu<T, NDIM>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, params, entire_idx_cvtr, sliced_idx_cvtr, entire, sliced);
}

template<int NDIM>
void LaunchSliceForward(DeviceCtx* ctx, const SliceParams& params, const float16* entire,
                        float16* sliced) {
  CHECK_EQ(params.ndim, NDIM);
  int64_t elem_cnt = params.elem_cnt();
  SliceIndexHelper<NDIM> entire_idx_cvtr(params.dims);
  SliceIndexHelper<NDIM> sliced_idx_cvtr(params.size);
  SliceForwardGpuHalf<NDIM>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, params, entire_idx_cvtr, sliced_idx_cvtr, reinterpret_cast<const half*>(entire),
          reinterpret_cast<half*>(sliced));
}

template<int NDIM>
void LaunchSliceBackward(DeviceCtx* ctx, const SliceParams& params, const float16* sliced,
                         float16* entire) {
  CHECK_EQ(params.ndim, NDIM);
  int64_t elem_cnt = params.elem_cnt();
  SliceIndexHelper<NDIM> entire_idx_cvtr(params.dims);
  SliceIndexHelper<NDIM> sliced_idx_cvtr(params.size);
  SliceBackwardGpuHalf<NDIM>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, params, entire_idx_cvtr, sliced_idx_cvtr, reinterpret_cast<half*>(entire),
          reinterpret_cast<const half*>(sliced));
}

template<typename T>
struct SliceSwitchUtil final {
#define MAKE_SLICE_SWITCH_ENTRY(func_name, N) func_name<T, N>
#define DEFINE_SLICE_SWITCH_UTIL_STATIC_METHOD(func_name) \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_SLICE_SWITCH_ENTRY, MAKE_NDIM_CTRV_SEQ(DIM_SEQ));

  DEFINE_SLICE_SWITCH_UTIL_STATIC_METHOD(LaunchSliceForward);
  DEFINE_SLICE_SWITCH_UTIL_STATIC_METHOD(LaunchSliceBackward);
#undef DEFINE_SLICE_SWITCH_UTIL_STATIC_METHOD
#undef MAKE_SLICE_SWITCH_ENTRY
};

template<>
struct SliceSwitchUtil<float16> {
#define MAKE_SLICE_SWITCH_ENTRY(func_name, N) func_name<N>
#define DEFINE_SLICE_SWITCH_UTIL_STATIC_METHOD(func_name) \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_SLICE_SWITCH_ENTRY, MAKE_NDIM_CTRV_SEQ(DIM_SEQ));

  DEFINE_SLICE_SWITCH_UTIL_STATIC_METHOD(LaunchSliceForward);
  DEFINE_SLICE_SWITCH_UTIL_STATIC_METHOD(LaunchSliceBackward);
#undef DEFINE_SLICE_SWITCH_UTIL_STATIC_METHOD
#undef MAKE_SLICE_SWITCH_ENTRY
};

}  // namespace

template<typename T>
struct SliceKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const SliceParams& params, const T* entire, T* sliced) {
    SliceParams fold_slice_params = FoldContiguousFullSliceDimensions(params);
    SliceSwitchUtil<T>::SwitchLaunchSliceForward(SwitchCase(fold_slice_params.ndim), ctx,
                                                 fold_slice_params, entire, sliced);
  }

  static void Backward(DeviceCtx* ctx, const SliceParams& params, const T* sliced, T* entire) {
    SliceParams fold_slice_params = FoldContiguousFullSliceDimensions(params);
    SliceSwitchUtil<T>::SwitchLaunchSliceBackward(SwitchCase(fold_slice_params.ndim), ctx,
                                                  fold_slice_params, sliced, entire);
  }
};

INSTANTIATE_SLICE_KERNEL_UTIL_WITH_DEVICE(DeviceType::kGPU)
INSTANTIATE_SLICE_KERNEL_UTIL(DeviceType::kGPU, float16)

}  // namespace oneflow
