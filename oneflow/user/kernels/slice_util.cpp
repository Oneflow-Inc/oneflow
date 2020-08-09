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

using SliceIndexConverter = NdIndexOffsetHelper<int64_t, kSliceMaxDims>;

int64_t SliceOffsetToEntireOffset(int64_t offset, const SliceParams& params,
                                  const SliceIndexConverter& entire_idx_cvtr,
                                  const SliceIndexConverter& sliced_idx_cvtr) {
  int64_t nd_index[kSliceMaxDims] = {0};
  sliced_idx_cvtr.OffsetToNdIndex(offset, nd_index, params.ndim);
  for (int64_t i = 0; i < params.ndim; ++i) {
    nd_index[i] = params.start[i] + params.step[i] * nd_index[i];
    assert(nd_index[i] >= 0);
    assert(nd_index[i] < params.dims[i]);
  }
  return entire_idx_cvtr.NdIndexToOffset(nd_index, params.ndim);
}

}  // namespace

void FoldContiguousFullSliceDimensions(SliceParams* params) {
  int cur_dim = 0;
  bool full_slice_on_prev_axis = false;
  FOR_RANGE(int, i, 0, params->ndim) {
    bool full_slice_on_cur_axis = params->IsFullSlice(i);
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
    full_slice_on_prev_axis = full_slice_on_cur_axis;
  }
  params->ndim = cur_dim + 1;
}

template<typename T>
struct SliceKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, SliceParams* params, const T* entire, T* sliced) {
    int64_t elem_cnt = params->elem_cnt();
    SliceIndexConverter entire_idx_cvtr(params->dims, params->ndim);
    SliceIndexConverter sliced_idx_cvtr(params->size, params->ndim);
    FOR_RANGE(int, i, 0, elem_cnt) {
      int64_t offset = SliceOffsetToEntireOffset(i, *params, entire_idx_cvtr, sliced_idx_cvtr);
      sliced[i] = entire[offset];
    }
  }

  static void Backward(DeviceCtx* ctx, SliceParams* params, const T* sliced, T* entire) {
    int64_t elem_cnt = params->elem_cnt();
    SliceIndexConverter entire_idx_cvtr(params->dims, params->ndim);
    SliceIndexConverter sliced_idx_cvtr(params->size, params->ndim);
    FOR_RANGE(int, i, 0, elem_cnt) {
      int64_t offset = SliceOffsetToEntireOffset(i, *params, entire_idx_cvtr, sliced_idx_cvtr);
      entire[offset] = sliced[i];
    }
  }
};

INSTANTIATE_SLICE_KERNEL_UTIL_WITH_DEVICE(DeviceType::kCPU)

}  // namespace oneflow
