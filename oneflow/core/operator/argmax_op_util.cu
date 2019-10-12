#include <cub/cub.cuh>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/switch_func.h"
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

template<typename T>
size_t InferTempStorageForCUBArgmax(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = -1;
  cub::CountingInputIterator<int32_t> counting_iter(0);
  cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

  cudaStream_t cuda_stream;
  CudaCheck(cudaStreamCreate(&cuda_stream));
  auto err = cub::DeviceSegmentedReduce::ArgMax(
      /* d_temp_storage */ static_cast<void*>(NULL),
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_in */ static_cast<T*>(NULL),
      /* d_out */ static_cast<cub::KeyValuePair<int32_t, T>*>(NULL),
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* stream */ cuda_stream);
  CudaCheck(err);
  CudaCheck(cudaStreamDestroy(cuda_stream));

  return temp_storage_bytes;
}

struct InferTempStorageForCUBArgmaxSwitchUtil final {
#define MAKE_INFER_TEMP_STORAGE_FOR_CUB_ARGMAX_SWITCH_ENTRY(func_name, KeyType) func_name<KeyType>
#define DEFINE_INFER_TEMP_STORAGE_FOR_CUB_ARGMAX_STATIC_SWITCH_FUNC(func_name)   \
  DEFINE_STATIC_SWITCH_FUNC(size_t, func_name,                                   \
                            MAKE_INFER_TEMP_STORAGE_FOR_CUB_ARGMAX_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
  DEFINE_INFER_TEMP_STORAGE_FOR_CUB_ARGMAX_STATIC_SWITCH_FUNC(InferTempStorageForCUBArgmax);
#undef DEFINE_INFER_TEMP_STORAGE_FOR_CUB_ARGMAX_STATIC_SWITCH_FUNC
#undef MAKE_INFER_TEMP_STORAGE_FOR_CUB_ARGMAX_SWITCH_ENTRY
};

size_t InferTempStorageForCUBArgmaxAtCompile(int32_t num_row, int32_t num_col, DataType data_type) {
  InferTempStorageForCUBArgmaxSwitchUtil::SwitchInferTempStorageForCUBArgmax(SwitchCase(data_type),
                                                                             num_row, num_col);
}

}  // namespace oneflow
