#include "oneflow/core/kernel/radix_sort_util.cuh"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/switch_func.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<typename T>
size_t InferTempStorage(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = -1;

  cub::CountingInputIterator<int32_t> counting_iter(0);
  cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

  cudaStream_t cuda_stream;
  CudaCheck(cudaStreamCreate(&cuda_stream));

  auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      /* d_temp_storage */ static_cast<void*>(NULL),
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_keys_in */ static_cast<T*>(NULL),
      /* d_keys_out */ static_cast<T*>(NULL),
      /* d_values_in */ static_cast<int32_t*>(NULL),
      /* d_values_out */ static_cast<int32_t*>(NULL),
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(T) * 8,
      /* stream */ cuda_stream);
  CudaCheck(err);

  CudaCheck(cudaStreamDestroy(cuda_stream));

  return temp_storage_bytes;
}

struct InferTempStorageSwitchUtil final {
#define MAKE_INFER_TEMP_STORAGE_SWITCH_ENTRY(func_name, T) func_name<T>
#define DEFINE_INFER_TEMP_STORAGE_STATIC_SWITCH_FUNC(func_name)                      \
  DEFINE_STATIC_SWITCH_FUNC(size_t, func_name, MAKE_INFER_TEMP_STORAGE_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
  DEFINE_INFER_TEMP_STORAGE_STATIC_SWITCH_FUNC(InferTempStorage);
#undef DEFINE_INFER_TEMP_STORAGE_STATIC_SWITCH_FUNC
#undef MAKE_INFER_TEMP_STORAGE_SWITCH_ENTRY
};

}  // namespace

size_t InferTempStorageForRadixSort(int32_t num_row, int32_t num_col, DataType data_type) {
  InferTempStorageSwitchUtil::SwitchInferTempStorage(SwitchCase(data_type), num_row, num_col);
}

}  // namespace oneflow
