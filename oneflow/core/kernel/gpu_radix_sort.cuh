#ifndef ONEFLOW_CORE_KERNEL_GPU_RADIX_SORT_CUH_
#define ONEFLOW_CORE_KERNEL_GPU_RADIX_SORT_CUH_

#include <cub/cub.cuh>
#include <glog/logging.h>
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace {

class SegmentOffsetCreator final {
 public:
  SegmentOffsetCreator(int32_t num_col) : num_col_(num_col) {}
  __device__ int32_t operator()(int32_t idx) const { return idx * num_col_; }

 private:
  int32_t num_col_;
};

}  // namespace

// Sort key-value pairs in ascending order
template<typename KeyType, typename ValueType>
void SortPairsAscending(const KeyType* keys_ptr, const ValueType* values_ptr, int32_t num_row,
                        int32_t num_col, void* temp_storage_ptr, int32_t temp_storage_bytes,
                        KeyType* sorted_keys_ptr, ValueType* sorted_values_ptr,
                        cudaStream_t cuda_stream) {
  cub::CountingInputIterator<int32_t> counting_iter(0);
  cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

  size_t rt_inferred_temp_storage_bytes = -1;
  auto err = cub::DeviceSegmentedRadixSort::SortPairs(
      /* d_temp_storage */ static_cast<void*>(NULL),
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_keys_in */ static_cast<KeyType*>(NULL),
      /* d_keys_out */ static_cast<KeyType*>(NULL),
      /* d_values_in */ static_cast<ValueType*>(NULL),
      /* d_values_out */ static_cast<ValueType*>(NULL),
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);
  CHECK_LE(rt_inferred_temp_storage_bytes, temp_storage_bytes);

  err = cub::DeviceSegmentedRadixSort::SortPairs(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_keys_in */ keys_ptr,
      /* d_keys_out */ sorted_keys_ptr,
      /* d_values_in */ values_ptr,
      /* d_values_out */ sorted_values_ptr,
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);
}

// Sort key-value pairs in descending order
template<typename KeyType, typename ValueType>
void SortPairsDescending(const KeyType* keys_ptr, const ValueType* values_ptr, int32_t num_row,
                         int32_t num_col, void* temp_storage_ptr, int32_t temp_storage_bytes,
                         KeyType* sorted_keys_ptr, ValueType* sorted_values_ptr,
                         cudaStream_t cuda_stream) {
  cub::CountingInputIterator<int32_t> counting_iter(0);
  cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

  size_t rt_inferred_temp_storage_bytes = -1;
  auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      /* d_temp_storage */ static_cast<void*>(NULL),
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_keys_in */ static_cast<KeyType*>(NULL),
      /* d_keys_out */ static_cast<KeyType*>(NULL),
      /* d_values_in */ static_cast<ValueType*>(NULL),
      /* d_values_out */ static_cast<ValueType*>(NULL),
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);
  CHECK_LE(rt_inferred_temp_storage_bytes, temp_storage_bytes);

  err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_keys_in */ keys_ptr,
      /* d_keys_out */ sorted_keys_ptr,
      /* d_values_in */ values_ptr,
      /* d_values_out */ sorted_values_ptr,
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);
}

// Sort keys only in ascending order
template<typename KeyType>
void SortKeysAscending(const KeyType* keys_ptr, int32_t num_row, int32_t num_col,
                       void* temp_storage_ptr, int32_t temp_storage_bytes, KeyType* sorted_keys_ptr,
                       cudaStream_t cuda_stream) {
  cub::CountingInputIterator<int32_t> counting_iter(0);
  cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

  size_t rt_inferred_temp_storage_bytes = -1;
  auto err = cub::DeviceSegmentedRadixSort::SortKeys(
      /* d_temp_storage */ static_cast<void*>(NULL),
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_keys_in */ static_cast<KeyType*>(NULL),
      /* d_keys_out */ static_cast<KeyType*>(NULL),
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);
  CHECK_LE(rt_inferred_temp_storage_bytes, temp_storage_bytes);

  err = cub::DeviceSegmentedRadixSort::SortKeys(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_keys_in */ keys_ptr,
      /* d_keys_out */ sorted_keys_ptr,
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);
}

// Sort keys only in descending order
template<typename KeyType>
void SortKeysDescending(const KeyType* keys_ptr, int32_t num_row, int32_t num_col,
                        void* temp_storage_ptr, int32_t temp_storage_bytes,
                        KeyType* sorted_keys_ptr, cudaStream_t cuda_stream) {
  cub::CountingInputIterator<int32_t> counting_iter(0);
  cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

  size_t rt_inferred_temp_storage_bytes = -1;
  auto err = cub::DeviceSegmentedRadixSort::SortKeysDescending(
      /* d_temp_storage */ static_cast<void*>(NULL),
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_keys_in */ static_cast<KeyType*>(NULL),
      /* d_keys_out */ static_cast<KeyType*>(NULL),
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);
  CHECK_LE(rt_inferred_temp_storage_bytes, temp_storage_bytes);

  err = cub::DeviceSegmentedRadixSort::SortKeysDescending(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_keys_in */ keys_ptr,
      /* d_keys_out */ sorted_keys_ptr,
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_GPU_RADIX_SORT_CUH_
