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

template<typename T, typename K, typename IDX>
__global__ void UnsortedSegmentSumGpu(const IDX data_elem_cnt, const K* segment_ids,
                                      const IDX num_segment_ids, const T* data,
                                      const IDX num_segments, const IDX inner_dim_size, T* out,
                                      const IDX segment_id_offset) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, data_elem_cnt) {
    const T val = data[i];
    if (val != static_cast<T>(0)) {
      const int64_t out_offset = GetOutOffset<K, IDX>(i, segment_ids, num_segment_ids, num_segments,
                                                      inner_dim_size, segment_id_offset);
      if (out_offset >= 0) { gpu_atomic_add(out + out_offset, val); }
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

template<typename T, typename K>
struct UnsortedSegmentSumKernelUtil<DeviceType::kGPU, T, K> final {
  static void UnsortedSegmentSum(DeviceCtx* ctx, const K* segment_ids, const T* data,
                                 int64_t num_segment_ids, int64_t num_segments,
                                 int64_t outer_dim_size, int64_t inner_dim_size,
                                 int64_t segment_id_offset, T* out) {
    const int64_t data_elem_cnt = outer_dim_size * num_segment_ids * inner_dim_size;
    if (IsSafeUseIndex32(num_segment_ids, num_segments, outer_dim_size, inner_dim_size)) {
      UnsortedSegmentSumGpu<T, K, int32_t>
          <<<BlocksNum4ThreadsNum(data_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              data_elem_cnt, segment_ids, num_segment_ids, data, num_segments, inner_dim_size, out,
              segment_id_offset);
    } else {
      UnsortedSegmentSumGpu<T, K, int64_t>
          <<<BlocksNum4ThreadsNum(data_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              data_elem_cnt, segment_ids, num_segment_ids, data, num_segments, inner_dim_size, out,
              segment_id_offset);
    }
  }
};

template<typename K>
struct UnsortedSegmentSumKernelUtil<DeviceType::kGPU, float16, K> final {
  static void UnsortedSegmentSum(DeviceCtx* ctx, const K* segment_ids, const float16* data,
                                 int64_t num_segment_ids, int64_t num_segments,
                                 int64_t outer_dim_size, int64_t inner_dim_size,
                                 int64_t segment_id_offset, float16* out) {
    UnsortedSegmentSumKernelUtil<DeviceType::kGPU, half, K>::UnsortedSegmentSum(
        ctx, segment_ids, reinterpret_cast<const half*>(data), num_segment_ids, num_segments,
        outer_dim_size, inner_dim_size, segment_id_offset, reinterpret_cast<half*>(out));
  }
};

#define INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_GPU(in_type_pair, index_type_pair)             \
  template struct UnsortedSegmentSumKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                               OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_GPU,
                                 UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_GPU

}  // namespace oneflow
