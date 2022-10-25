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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

SliceParams FoldContiguousFullSliceDimensions(const SliceParams& params) {
  SliceParams fold_slice_params;
  bool full_slice_on_prev_axis = false;
  FOR_RANGE(int, i, 0, params.ndim) {
    bool full_slice_on_cur_axis = params.IsFullSlice(i);
    if (full_slice_on_cur_axis && full_slice_on_prev_axis) {
      int cur_dim = fold_slice_params.ndim - 1;
      fold_slice_params.dims[cur_dim] *= params.dims[i];
      fold_slice_params.size[cur_dim] *= params.size[i];
    } else {
      int cur_dim = fold_slice_params.ndim;
      fold_slice_params.dims[cur_dim] = params.dims[i];
      fold_slice_params.start[cur_dim] = params.start[i];
      fold_slice_params.step[cur_dim] = params.step[i];
      fold_slice_params.size[cur_dim] = params.size[i];
      fold_slice_params.ndim += 1;
    }
    full_slice_on_prev_axis = full_slice_on_cur_axis;
  }
  return fold_slice_params;
}

template<typename T>
struct SliceKernelUtil<DeviceType::kCPU, T> {
  static void Forward(ep::Stream* stream, const SliceParams& params, const T* entire, T* sliced) {
    SliceParams fold_slice_params = FoldContiguousFullSliceDimensions(params);
    SwitchDoForward(SwitchCase(fold_slice_params.ndim), stream, fold_slice_params, entire, sliced);
  }

  static void Forward(ep::Stream* stream, const SliceParams& entire_params,
                      const SliceParams& sliced_params, const T* entire, T* sliced) {
    SwitchDoForward(SwitchCase(entire_params.ndim), stream, entire_params, sliced_params, entire,
                    sliced);
  }

  static void Backward(ep::Stream* stream, const SliceParams& params, const T* sliced, T* entire) {
    SliceParams fold_slice_params = FoldContiguousFullSliceDimensions(params);
    SwitchDoBackward(SwitchCase(fold_slice_params.ndim), stream, fold_slice_params, sliced, entire);
  }

 private:
  template<int NDIM>
  static void DoForward(ep::Stream* stream, const SliceParams& params, const T* entire, T* sliced) {
    CHECK_EQ(params.ndim, NDIM);
    int64_t elem_cnt = params.elem_cnt();
    SliceIndexHelper<NDIM> entire_idx_cvtr(params.dims);
    SliceIndexHelper<NDIM> sliced_idx_cvtr(params.size);
    MultiThreadLoop(elem_cnt, [&](int64_t i) {
      int64_t offset = SliceOffsetToEntireOffset<NDIM>(i, params, entire_idx_cvtr, sliced_idx_cvtr);
      sliced[i] = entire[offset];
    });
  }

  template<typename DoEachT>
  static void SteppedMultiThreadLoop(size_t elem_cnt, size_t step, const DoEachT& DoEach) {
    if (elem_cnt == 0) { return; }
    CHECK_GT(step, 0);
    CHECK_EQ(elem_cnt % step, 0);
    MultiThreadLoop(elem_cnt / step, [&](size_t i) { DoEach(i * step); });
  }

  template<int NDIM>
  static void DoForward(ep::Stream* stream, const SliceParams& entire_params,
                        const SliceParams& sliced_params, const T* entire, T* sliced) {
    CHECK_EQ(entire_params.ndim, NDIM);
    CHECK_EQ(sliced_params.ndim, NDIM);
    int64_t elem_cnt = entire_params.elem_cnt();
    SliceIndexHelper<NDIM> entire_splitted_large_idx_cvtr =
        NdIndexStrideOffsetHelper<int64_t, NDIM>(entire_params.stride);
    SliceIndexHelper<NDIM> sliced_splitted_large_idx_cvtr(entire_params.size);
    SliceIndexHelper<NDIM> entire_full_small_idx_cvtr =
        NdIndexStrideOffsetHelper<int64_t, NDIM>(sliced_params.stride);
    SliceIndexHelper<NDIM> sliced_full_small_idx_cvtr(sliced_params.size);

    int cnt = 1;
    int entire_target_stride = 1;
    int sliced_target_stride = 1;
    // Calculate the length of continuous part
    for (int i = NDIM - 1; i >= 0; i--) {
      if (entire_params.stride[i] != entire_target_stride
          || sliced_params.stride[i] != sliced_target_stride) {
        break;
      }
      entire_target_stride *= entire_params.size[i];
      sliced_target_stride *= sliced_params.size[i];
      if (sliced_params.step[i] == 1 && entire_params.step[i] == 1) {
        cnt *= sliced_params.size[i];
      }
      if (!entire_params.IsFullSlice(i) || !sliced_params.IsFullSlice(i)) { break; }
    }
    SteppedMultiThreadLoop(elem_cnt, cnt, [&](int64_t i) {
      const int64_t entire_offset = SliceOffsetToEntireOffset<NDIM>(
          i, entire_params, entire_splitted_large_idx_cvtr, sliced_splitted_large_idx_cvtr);
      const int64_t sliced_offset = SliceOffsetToEntireOffset<NDIM>(
          i, sliced_params, entire_full_small_idx_cvtr, sliced_full_small_idx_cvtr);
      std::copy(entire + entire_offset, entire + entire_offset + cnt, sliced + sliced_offset);
    });
  }

  template<int NDIM>
  static void DoBackward(ep::Stream* stream, const SliceParams& params, const T* sliced,
                         T* entire) {
    CHECK_EQ(params.ndim, NDIM);
    int64_t elem_cnt = params.elem_cnt();
    SliceIndexHelper<NDIM> entire_idx_cvtr(params.dims);
    SliceIndexHelper<NDIM> sliced_idx_cvtr(params.size);
    MultiThreadLoop(elem_cnt, [&](int64_t i) {
      int64_t offset = SliceOffsetToEntireOffset<NDIM>(i, params, entire_idx_cvtr, sliced_idx_cvtr);
      entire[offset] = sliced[i];
    });
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
INSTANTIATE_SLICE_KERNEL_UTIL(DeviceType::kCPU, bfloat16)

}  // namespace oneflow
