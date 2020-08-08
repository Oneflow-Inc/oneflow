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

SliceParams ConstructSliceParams(user_op::KernelComputeContext* ctx, const user_op::Tensor* entire,
                                 const user_op::Tensor* sliced) {
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  const int64_t ndim = entire->shape().NumAxes();
  CHECK_LE(ndim, kSliceMaxDims);
  CHECK_EQ(sliced->shape().NumAxes(), ndim);
  CHECK_EQ(start_vec.size(), ndim);
  CHECK_EQ(stop_vec.size(), ndim);
  CHECK_EQ(step_vec.size(), ndim);

  SliceParams params;
  std::memset(&params, 0, sizeof(SliceParams));
  // collapse contiguous dims who slice fully,
  // that it can reduce params.ndim thus reduce loop numbers in cuda kernel
  bool full_slice_on_prev_axis = false;
  FOR_RANGE(int, i, 0, ndim) {
    const int64_t dim_size = entire->shape().At(i);
    const int64_t slice_size = sliced->shape().At(i);
    const int64_t step = step_vec.at(i);
    CHECK_NE(step, 0);
    const int64_t start = RegulateSliceStart(start_vec.at(i), dim_size);
    const int64_t stop = RegulateSliceStop(stop_vec.at(i), dim_size);
    if (step > 0) {
      CHECK_LT(start + step * (slice_size - 1), stop);
    } else {
      CHECK_GT(start + step * (slice_size - 1), stop);
    }
    // full slice dim can be collapsed to prev full slice dim
    bool full_slice_on_cur_axis = IsFullSlice(start, stop, step, dim_size, false);
    if (i != 0 && full_slice_on_cur_axis && full_slice_on_prev_axis) {
      int cur_dim = params.ndim - 1;
      params.dims[cur_dim] *= dim_size;
      params.size[cur_dim] *= slice_size;
    } else {
      int cur_dim = params.ndim;
      params.dims[cur_dim] = dim_size;
      params.start[cur_dim] = start;
      params.step[cur_dim] = step;
      params.size[cur_dim] = slice_size;
      params.ndim += 1;
    }
    full_slice_on_prev_axis = full_slice_on_cur_axis;
  }
  return params;
}

template<typename T>
struct SliceKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const SliceParams& params, const T* entire, T* sliced) {
    int64_t elem_cnt = 1;
    FOR_RANGE(int, i, 0, params.ndim) { elem_cnt *= params.size[i]; }
    SliceIndexHelper entire_idx_cvtr(params.dims, params.ndim);
    SliceIndexHelper sliced_idx_cvtr(params.size, params.ndim);
    int64_t nd_index[kSliceMaxDims] = {0};
    FOR_RANGE(int, i, 0, elem_cnt) {
      int64_t offset =
          SliceOffsetToEntireOffset(i, nd_index, params, entire_idx_cvtr, sliced_idx_cvtr);
      sliced[i] = entire[offset];
    }
  }

  static void Backward(DeviceCtx* ctx, const SliceParams& params, const T* sliced, T* entire) {
    int64_t elem_cnt = 1;
    FOR_RANGE(int, i, 0, params.ndim) { elem_cnt *= params.size[i]; }
    SliceIndexHelper entire_idx_cvtr(params.dims, params.ndim);
    SliceIndexHelper sliced_idx_cvtr(params.size, params.ndim);
    int64_t nd_index[kSliceMaxDims] = {0};
    FOR_RANGE(int, i, 0, elem_cnt) {
      int64_t offset =
          SliceOffsetToEntireOffset(i, nd_index, params, entire_idx_cvtr, sliced_idx_cvtr);
      entire[offset] = sliced[i];
    }
  }
};

INSTANTIATE_SLICE_KERNEL_UTIL_WITH_DEVICE(DeviceType::kCPU)

}  // namespace oneflow
