#include "oneflow/core/kernel/radix_sort_util.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace {

class SegmentOffsetCreator final {
 public:
  SegmentOffsetCreator(int32_t num_col) : num_col_(num_col) {}
  __device__ int32_t operator()(int32_t idx) const { return idx * num_col_; }

 private:
  int32_t num_col_;
};

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

}  // namespace

// Infer temp storage for sorting key-value pairs in ascending order at compile stage
// FIX ME: should also pass value_data_type param here, use int64_t for now
size_t InferTempStorageForSortingPairsAscendingAtCompile(int32_t num_row, int32_t num_col,
                                                         DataType key_data_type) {
  InferTempStorageForSortingPairsAscendingSwitchUtil::
      SwitchInferTempStorageForSortingPairsAscending(SwitchCase(key_data_type), num_row, num_col);
}

// Infer temp storage for sorting key-value pairs in descending order at compile stage
// FIX ME: should also pass value_data_type param here, use int64_t for now
size_t InferTempStorageForSortingPairsDescendingAtCompile(int32_t num_row, int32_t num_col,
                                                          DataType key_data_type) {
  InferTempStorageForSortingPairsDescendingSwitchUtil::
      SwitchInferTempStorageForSortingPairsDescending(SwitchCase(key_data_type), num_row, num_col);
}

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

#define MAKE_SORT_PAIRS_ASCENDING_ENTRY(key_type_pair, value_type_pair)                       \
  template void                                                                               \
  SortPairsAscending<OF_PP_PAIR_FIRST(key_type_pair), OF_PP_PAIR_FIRST(value_type_pair)>(     \
      const OF_PP_PAIR_FIRST(key_type_pair) * keys_ptr,                                       \
      const OF_PP_PAIR_FIRST(value_type_pair) * values_ptr, int32_t num_row, int32_t num_col, \
      void* temp_storage_ptr, int32_t temp_storage_bytes,                                     \
      OF_PP_PAIR_FIRST(key_type_pair) * sorted_keys_ptr,                                      \
      OF_PP_PAIR_FIRST(value_type_pair) * sorted_values_ptr, cudaStream_t cuda_stream);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_SORT_PAIRS_ASCENDING_ENTRY, ARITHMETIC_DATA_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

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

#define MAKE_SORT_PAIRS_DESCENDING_ENTRY(key_type_pair, value_type_pair)                      \
  template void                                                                               \
  SortPairsDescending<OF_PP_PAIR_FIRST(key_type_pair), OF_PP_PAIR_FIRST(value_type_pair)>(    \
      const OF_PP_PAIR_FIRST(key_type_pair) * keys_ptr,                                       \
      const OF_PP_PAIR_FIRST(value_type_pair) * values_ptr, int32_t num_row, int32_t num_col, \
      void* temp_storage_ptr, int32_t temp_storage_bytes,                                     \
      OF_PP_PAIR_FIRST(key_type_pair) * sorted_keys_ptr,                                      \
      OF_PP_PAIR_FIRST(value_type_pair) * sorted_values_ptr, cudaStream_t cuda_stream);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_SORT_PAIRS_DESCENDING_ENTRY, ARITHMETIC_DATA_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

// Sort keys only in ascending order
template<typename KeyType>
void SortKeys() {
  // TODO
}

// Sort keys only in descending order
template<typename KeyType>
void SortKeysDescending() {
  // TODO
}

}  // namespace oneflow
