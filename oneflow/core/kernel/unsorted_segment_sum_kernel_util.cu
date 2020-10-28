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
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/kernel/unsorted_segment_sum_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/kernel.h"
#include <assert.h>

namespace oneflow {

namespace {

template<typename K, typename IDX>
__device__ IDX GetOutOffset(const IDX data_offset, const K* segment_ids, const IDX num_segment_ids,
                            const IDX num_segments, const IDX inner_dim_size,
                            const IDX segment_id_offset) {
  const IDX outer_dim_elem_cnt = num_segment_ids * inner_dim_size;
  const IDX outer_idx = data_offset / outer_dim_elem_cnt;
  const IDX segment_id_idx = data_offset % outer_dim_elem_cnt / inner_dim_size;
  const IDX inner_idx = data_offset % inner_dim_size;
  const K origin_idx = segment_ids[segment_id_idx];
  assert(origin_idx >= 0);
  const IDX idx = origin_idx - segment_id_offset;
  if (idx >= 0 && idx < num_segments) {
    return outer_idx * num_segments * inner_dim_size + idx * inner_dim_size + inner_idx;
  } else {
    return -1;
  }
}

template<typename T, typename K, typename IDX, typename U>
__global__ void UnsortedSegmentSumGpu(const IDX data_elem_cnt,
                                      const NdIndexOffsetHelper<IDX, 3> in_helper,
                                      const NdIndexOffsetHelper<IDX, 3> out_helper, const T* data,
                                      const K* segment_ids, const IDX num_segments,
                                      const IDX segment_id_offset, U* out) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, data_elem_cnt) {
    const T val = data[i];
    if (val != static_cast<T>(0)) {
      IDX outer_idx, segment_id_idx, inner_idx;
      in_helper.OffsetToNdIndex(i, outer_idx, segment_id_idx, inner_idx);
      const K origin_idx = segment_ids[segment_id_idx];
      assert(origin_idx >= 0);
      const IDX idx = origin_idx - segment_id_offset;
      if (idx >= 0 && idx < num_segments) {
        const int64_t out_offset = out_helper.NdIndexToOffset(outer_idx, idx, inner_idx);
        if (out_offset >= 0) { gpu_atomic_add(out + out_offset, static_cast<U>(val)); }
      }
    }
  }
}

bool IsSafeUseIndex32(const int64_t num_segment_ids, const int64_t num_segments,
                      const int64_t outer_dim_size, const int64_t inner_dim_size) {
  const int64_t data_elem_cnt = outer_dim_size * num_segment_ids * inner_dim_size;
  const int64_t out_elem_cnt = outer_dim_size * num_segments * inner_dim_size;
  return std::max(out_elem_cnt, data_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

}  // namespace

template<typename T, typename K, typename U>
struct UnsortedSegmentSumKernelUtil<DeviceType::kGPU, T, K, U> final {
  static void UnsortedSegmentSum(DeviceCtx* ctx, const K* segment_ids, const T* data,
                                 int64_t num_segment_ids, int64_t num_segments,
                                 int64_t outer_dim_size, int64_t inner_dim_size,
                                 int64_t segment_id_offset, U* out) {
    const int64_t data_elem_cnt = num_segment_ids * outer_dim_size * inner_dim_size;
    const int64_t out_elem_cnt = outer_dim_size * num_segments * inner_dim_size;

    if (std::max(out_elem_cnt, data_elem_cnt) < GetMaxVal<int32_t>() / 2) {
      NdIndexOffsetHelper<int32_t, 3> in_helper(outer_dim_size, num_segment_ids, inner_dim_size);
      NdIndexOffsetHelper<int32_t, 3> out_helper(outer_dim_size, num_segments, inner_dim_size);
      UnsortedSegmentSumGpu<T, K, int32_t, U>
          <<<BlocksNum4ThreadsNum(data_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              data_elem_cnt, in_helper, out_helper, data, segment_ids, num_segments,
              segment_id_offset, out);
    } else {
      NdIndexOffsetHelper<int64_t, 3> in_helper(outer_dim_size, num_segment_ids, inner_dim_size);
      NdIndexOffsetHelper<int64_t, 3> out_helper(outer_dim_size, num_segments, inner_dim_size);
      UnsortedSegmentSumGpu<T, K, int64_t, U>
          <<<BlocksNum4ThreadsNum(data_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              data_elem_cnt, in_helper, out_helper, data, segment_ids, num_segments,
              segment_id_offset, out);
    }
  }
};

template<typename K>
struct UnsortedSegmentSumKernelUtil<DeviceType::kGPU, float16, K, float16> final {
  static void UnsortedSegmentSum(DeviceCtx* ctx, const K* segment_ids, const float16* data,
                                 int64_t num_segment_ids, int64_t num_segments,
                                 int64_t outer_dim_size, int64_t inner_dim_size,
                                 int64_t segment_id_offset, float16* out) {
    UnsortedSegmentSumKernelUtil<DeviceType::kGPU, half, K, half>::UnsortedSegmentSum(
        ctx, segment_ids, reinterpret_cast<const half*>(data), num_segment_ids, num_segments,
        outer_dim_size, inner_dim_size, segment_id_offset, reinterpret_cast<half*>(out));
  }
};

#define INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_GPU(in_type_pair, index_type_pair)             \
  template struct UnsortedSegmentSumKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                               OF_PP_PAIR_FIRST(index_type_pair),                \
                                               OF_PP_PAIR_FIRST(in_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_GPU,
                                 UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_GPU

}  // namespace oneflow
