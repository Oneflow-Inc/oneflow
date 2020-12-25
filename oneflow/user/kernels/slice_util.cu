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

template<typename T>
size_t GetPackSize(const SliceParams& params, const T* entire, const T* sliced) {
  CHECK_GT(params.ndim, 0);
  const int64_t last_dim = params.ndim - 1;
  const int64_t mask = (params.dims[last_dim] * sizeof(T)) | (params.start[last_dim] * sizeof(T))
                       | (params.size[last_dim] * sizeof(T))
                       | static_cast<int64_t>(reinterpret_cast<uintptr_t>(entire))
                       | static_cast<int64_t>(reinterpret_cast<uintptr_t>(sliced));
  if ((mask & 0xF) == 0) {
    return 16;
  } else if ((mask & 0x7) == 0) {
    return 8;
  } else if ((mask & 0x3) == 0) {
    return 4;
  } else if ((mask & 0x1) == 0) {
    return 2;
  } else {
    return 1;
  }
}

template<typename T>
void GetPackedParams(const SliceParams& params, const T* entire, const T* sliced, size_t* pack_size,
                     SliceParams* packed_params) {
  CHECK_GT(params.ndim, 0);
  const int64_t last_dim = params.ndim - 1;
  if (params.step[last_dim] == 1) {
    *pack_size = GetPackSize<T>(params, entire, sliced);
    CHECK_GE(*pack_size, sizeof(T));
    const int64_t elem_per_pack = *pack_size / sizeof(T);
    *packed_params = params;
    packed_params->dims[last_dim] /= elem_per_pack;
    packed_params->start[last_dim] /= elem_per_pack;
    packed_params->size[last_dim] /= elem_per_pack;
  } else {
    *pack_size = sizeof(T);
    *packed_params = params;
  }
}

}  // namespace

template<typename T>
struct SliceKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const SliceParams& params, const T* entire, T* sliced) {
    SliceParams fold_slice_params = FoldContiguousFullSliceDimensions(params);
    size_t pack_size;
    SliceParams packed_params{};
    GetPackedParams<T>(fold_slice_params, entire, sliced, &pack_size, &packed_params);
    if (pack_size == 1) {
      SliceSwitchUtil<uint8_t>::SwitchLaunchSliceForward(
          SwitchCase(packed_params.ndim), ctx, packed_params,
          reinterpret_cast<const uint8_t*>(entire), reinterpret_cast<uint8_t*>(sliced));
    } else if (pack_size == 2) {
      SliceSwitchUtil<uint16_t>::SwitchLaunchSliceForward(
          SwitchCase(packed_params.ndim), ctx, packed_params,
          reinterpret_cast<const uint16_t*>(entire), reinterpret_cast<uint16_t*>(sliced));
    } else if (pack_size == 4) {
      SliceSwitchUtil<uint32_t>::SwitchLaunchSliceForward(
          SwitchCase(packed_params.ndim), ctx, packed_params,
          reinterpret_cast<const uint32_t*>(entire), reinterpret_cast<uint32_t*>(sliced));
    } else if (pack_size == 8) {
      SliceSwitchUtil<uint64_t>::SwitchLaunchSliceForward(
          SwitchCase(packed_params.ndim), ctx, packed_params,
          reinterpret_cast<const uint64_t*>(entire), reinterpret_cast<uint64_t*>(sliced));
    } else if (pack_size == 16) {
      SliceSwitchUtil<ulonglong2>::SwitchLaunchSliceForward(
          SwitchCase(packed_params.ndim), ctx, packed_params,
          reinterpret_cast<const ulonglong2*>(entire), reinterpret_cast<ulonglong2*>(sliced));
    } else {
      UNIMPLEMENTED();
    }
  }

  static void Backward(DeviceCtx* ctx, const SliceParams& params, const T* sliced, T* entire) {
    SliceParams fold_slice_params = FoldContiguousFullSliceDimensions(params);
    size_t pack_size;
    SliceParams packed_params{};
    GetPackedParams<T>(fold_slice_params, entire, sliced, &pack_size, &packed_params);
    if (pack_size == 1) {
      SliceSwitchUtil<uint8_t>::SwitchLaunchSliceBackward(
          SwitchCase(packed_params.ndim), ctx, packed_params,
          reinterpret_cast<const uint8_t*>(sliced), reinterpret_cast<uint8_t*>(entire));
    } else if (pack_size == 2) {
      SliceSwitchUtil<uint16_t>::SwitchLaunchSliceBackward(
          SwitchCase(packed_params.ndim), ctx, packed_params,
          reinterpret_cast<const uint16_t*>(sliced), reinterpret_cast<uint16_t*>(entire));
    } else if (pack_size == 4) {
      SliceSwitchUtil<uint32_t>::SwitchLaunchSliceBackward(
          SwitchCase(packed_params.ndim), ctx, packed_params,
          reinterpret_cast<const uint32_t*>(sliced), reinterpret_cast<uint32_t*>(entire));
    } else if (pack_size == 8) {
      SliceSwitchUtil<uint64_t>::SwitchLaunchSliceBackward(
          SwitchCase(packed_params.ndim), ctx, packed_params,
          reinterpret_cast<const uint64_t*>(sliced), reinterpret_cast<uint64_t*>(entire));
    } else if (pack_size == 16) {
      SliceSwitchUtil<ulonglong2>::SwitchLaunchSliceBackward(
          SwitchCase(packed_params.ndim), ctx, packed_params,
          reinterpret_cast<const ulonglong2*>(sliced), reinterpret_cast<ulonglong2*>(entire));
    } else {
      UNIMPLEMENTED();
    }
  }
};

INSTANTIATE_SLICE_KERNEL_UTIL_WITH_DEVICE(DeviceType::kGPU)
INSTANTIATE_SLICE_KERNEL_UTIL(DeviceType::kGPU, float16)

}  // namespace oneflow
