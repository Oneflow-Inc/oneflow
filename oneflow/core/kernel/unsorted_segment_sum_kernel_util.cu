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
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/kernel/kernel.h"
#include <assert.h>

namespace oneflow {

namespace {

template<typename T, typename K, typename IDX, typename U>
__global__ void UnsortedSegmentSumGpu(const IDX data_elem_cnt,
                                      const NdIndexOffsetHelper<IDX, 3> in_helper,
                                      const NdIndexOffsetHelper<IDX, 3> out_helper, const U* data,
                                      const K* segment_ids, const IDX num_segments,
                                      const IDX segment_id_offset, T* out) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, data_elem_cnt) {
    const U val = data[i];
    if (val != static_cast<U>(0)) {
      IDX outer_idx, segment_id_idx, inner_idx;
      in_helper.OffsetToNdIndex(i, outer_idx, segment_id_idx, inner_idx);
      const K origin_idx = segment_ids[segment_id_idx];
      assert(origin_idx >= 0);
      const IDX idx = origin_idx - segment_id_offset;
      if (idx >= 0 && idx < num_segments) {
        const int64_t out_offset = out_helper.NdIndexToOffset(outer_idx, idx, inner_idx);
        if (out_offset >= 0) { cuda::atomic::Add(out + out_offset, static_cast<T>(val)); }
      }
    }
  }
}

template<typename T, typename K, typename IDX, typename U>
__global__ void UnsortedSegmentColSumGpu(const IDX data_elem_cnt,
                                         const NdIndexOffsetHelper<IDX, 2> in_helper,
                                         const NdIndexOffsetHelper<IDX, 2> out_helper,
                                         const U* data, const K* segment_ids,
                                         const IDX num_segments, const IDX segment_id_offset,
                                         T* out) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, data_elem_cnt) {
    const U val = data[i];
    if (val != static_cast<U>(0)) {
      IDX outer_idx, segment_id_idx;
      in_helper.OffsetToNdIndex(i, outer_idx, segment_id_idx);
      const K origin_idx = segment_ids[segment_id_idx];
      assert(origin_idx >= 0);
      const IDX idx = origin_idx - segment_id_offset;
      if (idx >= 0 && idx < num_segments) {
        const int64_t out_offset = out_helper.NdIndexToOffset(outer_idx, idx);
        if (out_offset >= 0) { cuda::atomic::Add(out + out_offset, static_cast<T>(val)); }
      }
    }
  }
}

template<typename T, typename K, typename IDX, typename U>
__global__ void UnsortedSegmentRowSumGpu(const IDX data_elem_cnt,
                                         const NdIndexOffsetHelper<IDX, 2> in_helper,
                                         const NdIndexOffsetHelper<IDX, 2> out_helper,
                                         const U* data, const K* segment_ids,
                                         const IDX num_segments, const IDX segment_id_offset,
                                         T* out) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, data_elem_cnt) {
    const U val = data[i];
    if (val != static_cast<U>(0)) {
      IDX segment_id_idx, inner_idx;
      in_helper.OffsetToNdIndex(i, segment_id_idx, inner_idx);
      const K origin_idx = segment_ids[segment_id_idx];
      assert(origin_idx >= 0);
      const IDX idx = origin_idx - segment_id_offset;
      if (idx >= 0 && idx < num_segments) {
        const int64_t out_offset = out_helper.NdIndexToOffset(idx, inner_idx);
        if (out_offset >= 0) { cuda::atomic::Add(out + out_offset, static_cast<T>(val)); }
      }
    }
  }
}

template<typename T, typename K, typename IDX, typename U>
void UnsortedSegmentSumUtil(DeviceCtx* ctx, const K* segment_ids, const U* data,
                            IDX num_segment_ids, IDX num_segments, IDX outer_dim_size,
                            IDX inner_dim_size, IDX segment_id_offset, T* out) {
  const IDX data_elem_cnt = num_segment_ids * outer_dim_size * inner_dim_size;
  if (inner_dim_size == 1) {
    NdIndexOffsetHelper<IDX, 2> in_helper(outer_dim_size, num_segment_ids);
    NdIndexOffsetHelper<IDX, 2> out_helper(outer_dim_size, num_segments);
    UnsortedSegmentColSumGpu<T, K, IDX, U>
        <<<BlocksNum4ThreadsNum(data_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            data_elem_cnt, in_helper, out_helper, data, segment_ids, num_segments,
            segment_id_offset, out);

  } else if (outer_dim_size == 1) {
    NdIndexOffsetHelper<IDX, 2> in_helper(num_segment_ids, inner_dim_size);
    NdIndexOffsetHelper<IDX, 2> out_helper(num_segments, inner_dim_size);
    UnsortedSegmentRowSumGpu<T, K, IDX, U>
        <<<BlocksNum4ThreadsNum(data_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            data_elem_cnt, in_helper, out_helper, data, segment_ids, num_segments,
            segment_id_offset, out);

  } else {
    NdIndexOffsetHelper<IDX, 3> in_helper(outer_dim_size, num_segment_ids, inner_dim_size);
    NdIndexOffsetHelper<IDX, 3> out_helper(outer_dim_size, num_segments, inner_dim_size);
    UnsortedSegmentSumGpu<T, K, IDX, U>
        <<<BlocksNum4ThreadsNum(data_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            data_elem_cnt, in_helper, out_helper, data, segment_ids, num_segments,
            segment_id_offset, out);
  }
}

}  // namespace

template<typename T, typename K, typename U>
struct UnsortedSegmentSumKernelUtil<DeviceType::kGPU, T, K, U> final {
  static void UnsortedSegmentSum(DeviceCtx* ctx, const K* segment_ids, const U* data,
                                 int64_t num_segment_ids, int64_t num_segments,
                                 int64_t outer_dim_size, int64_t inner_dim_size,
                                 int64_t segment_id_offset, T* out) {
    const int64_t data_elem_cnt = num_segment_ids * outer_dim_size * inner_dim_size;
    const int64_t out_elem_cnt = outer_dim_size * num_segments * inner_dim_size;

    if (std::max(data_elem_cnt, out_elem_cnt) < GetMaxVal<int32_t>() / 2) {
      UnsortedSegmentSumUtil<T, K, int32_t, U>(ctx, segment_ids, data, num_segment_ids,
                                               num_segments, outer_dim_size, inner_dim_size,
                                               segment_id_offset, out);
    } else {
      UnsortedSegmentSumUtil<T, K, int64_t, U>(ctx, segment_ids, data, num_segment_ids,
                                               num_segments, outer_dim_size, inner_dim_size,
                                               segment_id_offset, out);
    }
  }
};

template<typename K>
struct UnsortedSegmentSumKernelUtil<DeviceType::kGPU, float, K, float16> final {
  static void UnsortedSegmentSum(DeviceCtx* ctx, const K* segment_ids, const float16* data,
                                 int64_t num_segment_ids, int64_t num_segments,
                                 int64_t outer_dim_size, int64_t inner_dim_size,
                                 int64_t segment_id_offset, float* out) {
    UnsortedSegmentSumKernelUtil<DeviceType::kGPU, float, K, half>::UnsortedSegmentSum(
        ctx, segment_ids, reinterpret_cast<const half*>(data), num_segment_ids, num_segments,
        outer_dim_size, inner_dim_size, segment_id_offset, out);
  }
};

#define INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_GPU(in_type_pair, index_type_pair)             \
  template struct UnsortedSegmentSumKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                               OF_PP_PAIR_FIRST(index_type_pair),                \
                                               OF_PP_PAIR_FIRST(in_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_GPU,
                                 UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_GPU

#define INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_HALF_GPU(in_type_pair, index_type_pair,             \
                                                      out_type_pair)                             \
  template struct UnsortedSegmentSumKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                               OF_PP_PAIR_FIRST(index_type_pair),                \
                                               OF_PP_PAIR_FIRST(out_type_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_HALF_GPU,
                                 OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat), INDEX_DATA_TYPE_SEQ,
                                 FLOAT16_DATA_TYPE_SEQ);

#undef INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_HALF_GPU

}  // namespace oneflow
