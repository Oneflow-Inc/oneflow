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
#ifndef ONEFLOW_USER_KERNELS_RADIX_SORT_CUH_
#define ONEFLOW_USER_KERNELS_RADIX_SORT_CUH_

#include <cub/cub.cuh>
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace {

class MultiplyFunctor final {
 public:
  MultiplyFunctor(int32_t num_col) : num_col_(num_col) {}
  __host__ __device__ __forceinline__ int32_t operator()(int32_t idx) const {
    return idx * num_col_;
  }

 private:
  int32_t num_col_;
};

}  // namespace

template<typename KeyType, typename ValueType>
size_t InferTempStorageForSortPairsAscending(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = 0;
  if (num_row > 1) {
    using SegmentOffsetIter =
        cub::TransformInputIterator<int32_t, MultiplyFunctor, cub::CountingInputIterator<int32_t>>;

    cub::CountingInputIterator<int32_t> counting_iter(0);
    MultiplyFunctor multiply_functor(num_col);
    SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

    auto err = cub::DeviceSegmentedRadixSort::SortPairs<KeyType, ValueType, SegmentOffsetIter>(
        /* d_temp_storage */ nullptr,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_keys_in */ nullptr,
        /* d_keys_out */ nullptr,
        /* d_values_in */ nullptr,
        /* d_values_out */ nullptr,
        /* num_items */ num_row * num_col,
        /* num_segments */ num_row,
        /* d_begin_offsets */ segment_offset_iter,
        /* d_end_offsets */ segment_offset_iter + 1,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ 0);
    OF_CUDA_CHECK(err);
  } else {
    auto err = cub::DeviceRadixSort::SortPairs<KeyType, ValueType>(
        /* d_temp_storage */ nullptr,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_keys_in */ nullptr,
        /* d_keys_out */ nullptr,
        /* d_values_in */ nullptr,
        /* d_values_out */ nullptr,
        /* num_items */ num_row * num_col,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ 0);
    OF_CUDA_CHECK(err);
  }

  return temp_storage_bytes;
}

template<typename KeyType, typename ValueType>
size_t InferTempStorageForSortPairsDescending(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = 0;
  if (num_row > 1) {
    using SegmentOffsetIter =
        cub::TransformInputIterator<int32_t, MultiplyFunctor, cub::CountingInputIterator<int32_t>>;

    cub::CountingInputIterator<int32_t> counting_iter(0);
    MultiplyFunctor multiply_functor(num_col);
    SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

    auto err =
        cub::DeviceSegmentedRadixSort::SortPairsDescending<KeyType, ValueType, SegmentOffsetIter>(
            /* d_temp_storage */ nullptr,
            /* temp_storage_bytes */ temp_storage_bytes,
            /* d_keys_in */ nullptr,
            /* d_keys_out */ nullptr,
            /* d_values_in */ nullptr,
            /* d_values_out */ nullptr,
            /* num_items */ num_row * num_col,
            /* num_segments */ num_row,
            /* d_begin_offsets */ segment_offset_iter,
            /* d_end_offsets */ segment_offset_iter + 1,
            /* begin_bit */ 0,
            /* end_bit */ sizeof(KeyType) * 8,
            /* stream */ 0);
    OF_CUDA_CHECK(err);
  } else {
    auto err = cub::DeviceRadixSort::SortPairsDescending<KeyType, ValueType>(
        /* d_temp_storage */ nullptr,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_keys_in */ nullptr,
        /* d_keys_out */ nullptr,
        /* d_values_in */ nullptr,
        /* d_values_out */ nullptr,
        /* num_items */ num_row * num_col,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ 0);
    OF_CUDA_CHECK(err);
  }

  return temp_storage_bytes;
}

template<typename KeyType>
size_t InferTempStorageForSortKeysAscending(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = 0;
  if (num_row > 1) {
    using SegmentOffsetIter =
        cub::TransformInputIterator<int32_t, MultiplyFunctor, cub::CountingInputIterator<int32_t>>;

    cub::CountingInputIterator<int32_t> counting_iter(0);
    MultiplyFunctor multiply_functor(num_col);
    SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

    auto err = cub::DeviceSegmentedRadixSort::SortKeys<KeyType, SegmentOffsetIter>(
        /* d_temp_storage */ nullptr,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_keys_in */ nullptr,
        /* d_keys_out */ nullptr,
        /* num_items */ num_row * num_col,
        /* num_segments */ num_row,
        /* d_begin_offsets */ segment_offset_iter,
        /* d_end_offsets */ segment_offset_iter + 1,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ 0);
    OF_CUDA_CHECK(err);
  } else {
    auto err = cub::DeviceRadixSort::SortKeys<KeyType>(
        /* d_temp_storage */ nullptr,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_keys_in */ nullptr,
        /* d_keys_out */ nullptr,
        /* num_items */ num_row * num_col,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ 0);
    OF_CUDA_CHECK(err);
  }
  return temp_storage_bytes;
}

template<typename KeyType>
size_t InferTempStorageForSortKeysDescending(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = 0;
  if (num_row > 1) {
    using SegmentOffsetIter =
        cub::TransformInputIterator<int32_t, MultiplyFunctor, cub::CountingInputIterator<int32_t>>;

    cub::CountingInputIterator<int32_t> counting_iter(0);
    MultiplyFunctor multiply_functor(num_col);
    SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

    auto err = cub::DeviceSegmentedRadixSort::SortKeysDescending<KeyType, SegmentOffsetIter>(
        /* d_temp_storage */ nullptr,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_keys_in */ nullptr,
        /* d_keys_out */ nullptr,
        /* num_items */ num_row * num_col,
        /* num_segments */ num_row,
        /* d_begin_offsets */ segment_offset_iter,
        /* d_end_offsets */ segment_offset_iter + 1,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ 0);
    OF_CUDA_CHECK(err);
  } else {
    auto err = cub::DeviceRadixSort::SortKeysDescending<KeyType>(
        /* d_temp_storage */ nullptr,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_keys_in */ nullptr,
        /* d_keys_out */ nullptr,
        /* num_items */ num_row * num_col,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ 0);
    OF_CUDA_CHECK(err);
  }

  return temp_storage_bytes;
}

template<typename KeyType, typename ValueType>
void SortPairsAscending(const KeyType* keys_ptr, const ValueType* values_ptr, int32_t num_row,
                        int32_t num_col, void* temp_storage_ptr, int32_t temp_storage_bytes,
                        KeyType* sorted_keys_ptr, ValueType* sorted_values_ptr,
                        cudaStream_t stream) {
  size_t rt_inferred_temp_storage_bytes =
      InferTempStorageForSortPairsAscending<KeyType, ValueType>(num_row, num_col);
  CHECK_LE(rt_inferred_temp_storage_bytes, temp_storage_bytes);
  if (num_row > 1) {
    using SegmentOffsetIter =
        cub::TransformInputIterator<int32_t, MultiplyFunctor, cub::CountingInputIterator<int32_t>>;

    cub::CountingInputIterator<int32_t> counting_iter(0);
    MultiplyFunctor multiply_functor(num_col);
    SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

    auto err = cub::DeviceSegmentedRadixSort::SortPairs(
        /* d_temp_storage */ temp_storage_ptr,
        /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
        /* d_keys_in */ keys_ptr,
        /* d_keys_out */ sorted_keys_ptr,
        /* d_values_in */ values_ptr,
        /* d_values_out */ sorted_values_ptr,
        /* num_items */ num_row * num_col,
        /* num_segments */ num_row,
        /* d_begin_offsets */ segment_offset_iter,
        /* d_end_offsets */ segment_offset_iter + 1,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ stream);
    OF_CUDA_CHECK(err);
  } else {
    auto err = cub::DeviceRadixSort::SortPairs(
        /* d_temp_storage */ temp_storage_ptr,
        /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
        /* d_keys_in */ keys_ptr,
        /* d_keys_out */ sorted_keys_ptr,
        /* d_values_in */ values_ptr,
        /* d_values_out */ sorted_values_ptr,
        /* num_items */ num_row * num_col,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ stream);
    OF_CUDA_CHECK(err);
  }
}

template<typename KeyType, typename ValueType>
void SortPairsDescending(const KeyType* keys_ptr, const ValueType* values_ptr, int32_t num_row,
                         int32_t num_col, void* temp_storage_ptr, int32_t temp_storage_bytes,
                         KeyType* sorted_keys_ptr, ValueType* sorted_values_ptr,
                         cudaStream_t stream) {
  size_t rt_inferred_temp_storage_bytes =
      InferTempStorageForSortPairsDescending<KeyType, ValueType>(num_row, num_col);
  CHECK_LE(rt_inferred_temp_storage_bytes, temp_storage_bytes);

  if (num_row > 1) {
    using SegmentOffsetIter =
        cub::TransformInputIterator<int32_t, MultiplyFunctor, cub::CountingInputIterator<int32_t>>;

    cub::CountingInputIterator<int32_t> counting_iter(0);
    MultiplyFunctor multiply_functor(num_col);
    SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        /* d_temp_storage */ temp_storage_ptr,
        /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
        /* d_keys_in */ keys_ptr,
        /* d_keys_out */ sorted_keys_ptr,
        /* d_values_in */ values_ptr,
        /* d_values_out */ sorted_values_ptr,
        /* num_items */ num_row * num_col,
        /* num_segments */ num_row,
        /* d_begin_offsets */ segment_offset_iter,
        /* d_end_offsets */ segment_offset_iter + 1,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ stream);
    OF_CUDA_CHECK(err);
  } else {
    auto err = cub::DeviceRadixSort::SortPairsDescending(
        /* d_temp_storage */ temp_storage_ptr,
        /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
        /* d_keys_in */ keys_ptr,
        /* d_keys_out */ sorted_keys_ptr,
        /* d_values_in */ values_ptr,
        /* d_values_out */ sorted_values_ptr,
        /* num_items */ num_row * num_col,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ stream);
    OF_CUDA_CHECK(err);
  }
}

template<typename KeyType>
void SortKeysAscending(const KeyType* keys_ptr, int32_t num_row, int32_t num_col,
                       void* temp_storage_ptr, int32_t temp_storage_bytes, KeyType* sorted_keys_ptr,
                       cudaStream_t stream) {
  size_t rt_inferred_temp_storage_bytes =
      InferTempStorageForSortKeysAscending<KeyType>(num_row, num_col);
  CHECK_LE(rt_inferred_temp_storage_bytes, temp_storage_bytes);

  if (num_row > 1) {
    using SegmentOffsetIter =
        cub::TransformInputIterator<int32_t, MultiplyFunctor, cub::CountingInputIterator<int32_t>>;

    cub::CountingInputIterator<int32_t> counting_iter(0);
    MultiplyFunctor multiply_functor(num_col);
    SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

    auto err = cub::DeviceSegmentedRadixSort::SortKeys(
        /* d_temp_storage */ temp_storage_ptr,
        /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
        /* d_keys_in */ keys_ptr,
        /* d_keys_out */ sorted_keys_ptr,
        /* num_items */ num_row * num_col,
        /* num_segments */ num_row,
        /* d_begin_offsets */ segment_offset_iter,
        /* d_end_offsets */ segment_offset_iter + 1,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ stream);
    OF_CUDA_CHECK(err);
  } else {
    auto err = cub::DeviceRadixSort::SortKeys(
        /* d_temp_storage */ temp_storage_ptr,
        /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
        /* d_keys_in */ keys_ptr,
        /* d_keys_out */ sorted_keys_ptr,
        /* num_items */ num_row * num_col,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ stream);
    OF_CUDA_CHECK(err);
  }
}

template<typename KeyType>
void SortKeysDescending(const KeyType* keys_ptr, int32_t num_row, int32_t num_col,
                        void* temp_storage_ptr, int32_t temp_storage_bytes,
                        KeyType* sorted_keys_ptr, cudaStream_t stream) {
  size_t rt_inferred_temp_storage_bytes =
      InferTempStorageForSortKeysDescending<KeyType>(num_row, num_col);
  CHECK_LE(rt_inferred_temp_storage_bytes, temp_storage_bytes);

  if (num_row > 1) {
    using SegmentOffsetIter =
        cub::TransformInputIterator<int32_t, MultiplyFunctor, cub::CountingInputIterator<int32_t>>;

    cub::CountingInputIterator<int32_t> counting_iter(0);
    MultiplyFunctor multiply_functor(num_col);
    SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

    auto err = cub::DeviceSegmentedRadixSort::SortKeysDescending(
        /* d_temp_storage */ temp_storage_ptr,
        /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
        /* d_keys_in */ keys_ptr,
        /* d_keys_out */ sorted_keys_ptr,
        /* num_items */ num_row * num_col,
        /* num_segments */ num_row,
        /* d_begin_offsets */ segment_offset_iter,
        /* d_end_offsets */ segment_offset_iter + 1,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ stream);
    OF_CUDA_CHECK(err);
  } else {
    auto err = cub::DeviceRadixSort::SortKeysDescending(
        /* d_temp_storage */ temp_storage_ptr,
        /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
        /* d_keys_in */ keys_ptr,
        /* d_keys_out */ sorted_keys_ptr,
        /* num_items */ num_row * num_col,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(KeyType) * 8,
        /* stream */ stream);
    OF_CUDA_CHECK(err);
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_RADIX_SORT_CUH_
