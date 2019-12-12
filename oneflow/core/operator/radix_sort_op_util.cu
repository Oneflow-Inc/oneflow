#include "oneflow/core/operator/radix_sort_op_util.h"
#include "oneflow/core/kernel/gpu_radix_sort.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

// FIX ME: should also has a template param for ValueType, use int64_t for now
template<typename KeyType>
size_t InferTempStorageForSortingPairsAscending(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = -1;

  cub::CountingInputIterator<int32_t> counting_iter(0);
  cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

  cudaStream_t cuda_stream;
  CudaCheck(cudaStreamCreate(&cuda_stream));

  auto err = cub::DeviceSegmentedRadixSort::SortPairs(
      /* d_temp_storage */ static_cast<void*>(NULL),
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_keys_in */ static_cast<KeyType*>(NULL),
      /* d_keys_out */ static_cast<KeyType*>(NULL),
      /* d_values_in */ static_cast<int64_t*>(NULL),
      /* d_values_out */ static_cast<int64_t*>(NULL),
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);
  CudaCheck(cudaStreamSynchronize(cuda_stream));
  if (temp_storage_bytes == 0) { temp_storage_bytes = 1; }
  CudaCheck(cudaStreamDestroy(cuda_stream));

  return temp_storage_bytes;
}

struct InferTempStorageForSortingPairsAscendingSwitchUtil final {
#define MAKE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_ASCENDING_SWITCH_ENTRY(func_name, KeyType) \
  func_name<KeyType>
#define DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_ASCENDING_STATIC_SWITCH_FUNC(func_name)   \
  DEFINE_STATIC_SWITCH_FUNC(size_t, func_name,                                                \
                            MAKE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_ASCENDING_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
  DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_ASCENDING_STATIC_SWITCH_FUNC(
      InferTempStorageForSortingPairsAscending);
#undef DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_ASCENDING_STATIC_SWITCH_FUNC
#undef MAKE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_ASCENDING_SWITCH_ENTRY
};

// FIX ME: should also has a template param for ValueType, use int64_t for now
template<typename KeyType>
size_t InferTempStorageForSortingPairsDescending(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = -1;

  cub::CountingInputIterator<int32_t> counting_iter(0);
  cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

  cudaStream_t cuda_stream;
  CudaCheck(cudaStreamCreate(&cuda_stream));

  auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      /* d_temp_storage */ static_cast<void*>(NULL),
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_keys_in */ static_cast<KeyType*>(NULL),
      /* d_keys_out */ static_cast<KeyType*>(NULL),
      /* d_values_in */ static_cast<int64_t*>(NULL),
      /* d_values_out */ static_cast<int64_t*>(NULL),
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);
  CudaCheck(cudaStreamSynchronize(cuda_stream));
  if (temp_storage_bytes == 0) { temp_storage_bytes = 1; }
  CudaCheck(cudaStreamDestroy(cuda_stream));

  return temp_storage_bytes;
}

struct InferTempStorageForSortingPairsDescendingSwitchUtil final {
#define MAKE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_DESCENDING_SWITCH_ENTRY(func_name, KeyType) \
  func_name<KeyType>
#define DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_DESCENDING_STATIC_SWITCH_FUNC(func_name)   \
  DEFINE_STATIC_SWITCH_FUNC(size_t, func_name,                                                 \
                            MAKE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_DESCENDING_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
  DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_DESCENDING_STATIC_SWITCH_FUNC(
      InferTempStorageForSortingPairsDescending);
#undef DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_DESCENDING_STATIC_SWITCH_FUNC
#undef MAKE_INFER_TEMP_STORAGE_FOR_SORTING_PAIRS_DESCENDING_SWITCH_ENTRY
};

template<typename KeyType>
size_t InferTempStorageForSortingKeysAscending(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = -1;

  cub::CountingInputIterator<int32_t> counting_iter(0);
  cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

  cudaStream_t cuda_stream;
  CudaCheck(cudaStreamCreate(&cuda_stream));

  auto err = cub::DeviceSegmentedRadixSort::SortKeys(
      /* d_temp_storage */ static_cast<void*>(NULL),
      /* temp_storage_bytes */ temp_storage_bytes,
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
  CudaCheck(cudaStreamSynchronize(cuda_stream));
  if (temp_storage_bytes == 0) { temp_storage_bytes = 1; }
  CudaCheck(cudaStreamDestroy(cuda_stream));

  return temp_storage_bytes;
}

struct InferTempStorageForSortingKeysAscendingSwitchUtil final {
#define MAKE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_ASCENDING_SWITCH_ENTRY(func_name, KeyType) \
  func_name<KeyType>
#define DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_ASCENDING_STATIC_SWITCH_FUNC(func_name)   \
  DEFINE_STATIC_SWITCH_FUNC(size_t, func_name,                                               \
                            MAKE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_ASCENDING_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
  DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_ASCENDING_STATIC_SWITCH_FUNC(
      InferTempStorageForSortingKeysAscending);
#undef DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_ASCENDING_STATIC_SWITCH_FUNC
#undef MAKE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_ASCENDING_SWITCH_ENTRY
};

template<typename KeyType>
size_t InferTempStorageForSortingKeysDescending(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = -1;

  cub::CountingInputIterator<int32_t> counting_iter(0);
  cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

  cudaStream_t cuda_stream;
  CudaCheck(cudaStreamCreate(&cuda_stream));

  auto err = cub::DeviceSegmentedRadixSort::SortKeysDescending(
      /* d_temp_storage */ static_cast<void*>(NULL),
      /* temp_storage_bytes */ temp_storage_bytes,
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
  CudaCheck(cudaStreamSynchronize(cuda_stream));
  if (temp_storage_bytes == 0) { temp_storage_bytes = 1; }
  CudaCheck(cudaStreamDestroy(cuda_stream));

  return temp_storage_bytes;
}

struct InferTempStorageForSortingKeysDescendingSwitchUtil final {
#define MAKE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_DESCENDING_SWITCH_ENTRY(func_name, KeyType) \
  func_name<KeyType>
#define DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_DESCENDING_STATIC_SWITCH_FUNC(func_name)   \
  DEFINE_STATIC_SWITCH_FUNC(size_t, func_name,                                                \
                            MAKE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_DESCENDING_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
  DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_DESCENDING_STATIC_SWITCH_FUNC(
      InferTempStorageForSortingKeysDescending);
#undef DEFINE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_DESCENDING_STATIC_SWITCH_FUNC
#undef MAKE_INFER_TEMP_STORAGE_FOR_SORTING_KEYS_DESCENDING_SWITCH_ENTRY
};

// Infer temp storage for sorting key-value pairs in ascending order at compile stage
// FIX ME: should also pass value_data_type param here, use int64_t for now
size_t InferTempStorageForSortingPairsAscendingAtCompile(int32_t num_row, int32_t num_col,
                                                         DataType key_data_type) {
  return InferTempStorageForSortingPairsAscendingSwitchUtil::
      SwitchInferTempStorageForSortingPairsAscending(SwitchCase(key_data_type), num_row, num_col);
}

// Infer temp storage for sorting key-value pairs in descending order at compile stage
// FIX ME: should also pass value_data_type param here, use int64_t for now
size_t InferTempStorageForSortingPairsDescendingAtCompile(int32_t num_row, int32_t num_col,
                                                          DataType key_data_type) {
  return InferTempStorageForSortingPairsDescendingSwitchUtil::
      SwitchInferTempStorageForSortingPairsDescending(SwitchCase(key_data_type), num_row, num_col);
}

// Infer temp storage for sorting keys in ascending order at compile stage
size_t InferTempStorageForSortingKeysAscendingAtCompile(int32_t num_row, int32_t num_col,
                                                        DataType key_data_type) {
  return InferTempStorageForSortingKeysAscendingSwitchUtil::
      SwitchInferTempStorageForSortingKeysAscending(SwitchCase(key_data_type), num_row, num_col);
}

// Infer temp storage for sorting keys in descending order at compile stage
size_t InferTempStorageForSortingKeysDescendingAtCompile(int32_t num_row, int32_t num_col,
                                                         DataType key_data_type) {
  return InferTempStorageForSortingKeysDescendingSwitchUtil::
      SwitchInferTempStorageForSortingKeysDescending(SwitchCase(key_data_type), num_row, num_col);
}

}  // namespace oneflow
