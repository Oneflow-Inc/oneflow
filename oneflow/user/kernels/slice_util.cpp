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

void FoldContiguousFullSliceDimensions(SliceParams* params) {
  int cur_dim = 0;
  bool full_slice_on_prev_axis = false;
  FOR_RANGE(int, i, 0, params->ndim) {
    bool full_slice_on_cur_axis = params->IsFullSlice(i);
    if (i != 0) {
      if (full_slice_on_cur_axis && full_slice_on_prev_axis) {
        params->dims[cur_dim] *= params->dims[i];
        params->size[cur_dim] *= params->size[i];
      } else {
        cur_dim += 1;
        params->dims[cur_dim] = params->dims[i];
        params->start[cur_dim] = params->start[i];
        params->step[cur_dim] = params->step[i];
        params->size[cur_dim] = params->size[i];
      }
    }
    full_slice_on_prev_axis = full_slice_on_cur_axis;
  }
  params->ndim = cur_dim + 1;
}

template<typename T>
struct SliceKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, SliceParams* params, const T* entire, T* sliced) {
    FoldContiguousFullSliceDimensions(params);
    SwitchDoForward(SwitchCase(params->ndim), ctx, *params, entire, sliced);
  }

  static void Backward(DeviceCtx* ctx, SliceParams* params, const T* sliced, T* entire) {
    FoldContiguousFullSliceDimensions(params);
    SwitchDoBackward(SwitchCase(params->ndim), ctx, *params, sliced, entire);
  }

 private:
  template<int NDIM>
  static void DoForward(DeviceCtx* ctx, const SliceParams& params, const T* entire, T* sliced) {
    CHECK_EQ(params.ndim, NDIM);
    int64_t elem_cnt = params.elem_cnt();
    SliceIndexHelper<NDIM> entire_idx_cvtr(params.dims);
    SliceIndexHelper<NDIM> sliced_idx_cvtr(params.size);
    FOR_RANGE(int, i, 0, elem_cnt) {
      int64_t offset = SliceOffsetToEntireOffset(i, params, entire_idx_cvtr, sliced_idx_cvtr);
      sliced[i] = entire[offset];
    }
  }

  template<int NDIM>
  static void DoBackward(DeviceCtx* ctx, const SliceParams& params, const T* sliced, T* entire) {
    CHECK_EQ(params.ndim, NDIM);
    int64_t elem_cnt = params.elem_cnt();
    SliceIndexHelper<NDIM> entire_idx_cvtr(params.dims);
    SliceIndexHelper<NDIM> sliced_idx_cvtr(params.size);
    FOR_RANGE(int, i, 0, elem_cnt) {
      int64_t offset = SliceOffsetToEntireOffset(i, params, entire_idx_cvtr, sliced_idx_cvtr);
      entire[offset] = sliced[i];
    }
  }

#define MAKE_SLICE_KERNEL_UTIL_SWITCH_ENTRY(func_name, N) \
  SliceKernelUtil<DeviceType::kCPU, T>::func_name<N>
#define DEFINE_SLICE_KERNEL_UTIL_SWITCH_STATIC_METHOD(func_name)                  \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_SLICE_KERNEL_UTIL_SWITCH_ENTRY, \
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));

  DEFINE_SLICE_KERNEL_UTIL_SWITCH_STATIC_METHOD(DoForward);
  DEFINE_SLICE_KERNEL_UTIL_SWITCH_STATIC_METHOD(DoBackward);
#undef DEFINE_SLICE_KERNEL_UTIL_SWITCH_STATIC_METHOD
#undef MAKE_SLICE_KERNEL_UTIL_SWITCH_ENTRY
};

INSTANTIATE_SLICE_KERNEL_UTIL_WITH_DEVICE(DeviceType::kCPU)

}  // namespace oneflow
