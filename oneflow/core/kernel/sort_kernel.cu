#include "oneflow/core/kernel/sort_kernel.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

class SegmentOffsetCreator {
 public:
  SegmentOffsetCreator(int32_t num_col) : num_col_(num_col) {}
  __device__ int32_t operator()(int32_t idx) const { return idx * num_col_; }

 private:
  int32_t num_col_;
};

}  // namespace

template<typename T>
struct SortUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const T* key_ptr, const int32_t* value_ptr,
                      void* temp_storage_ptr, size_t temp_storage_bytes, int32_t num_row,
                      int32_t num_col, T* sorted_key_ptr, int32_t* sorted_value_ptr) {
    cub::CountingInputIterator<int32_t> counting_iter(0);
    cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
        segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

    cudaStream_t cuda_stream = ctx->cuda_stream();

    // When d_temp_storage is NULL, no work is done and the required allocation size is returned in
    // temp_storage_bytes.
    size_t infered_temp_storage_byte_size = -1;
    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        /* d_temp_storage */ nullptr,
        /* temp_storage_bytes */ infered_temp_storage_byte_size,
        /* d_keys_in */ key_ptr,
        /* d_keys_out */ sorted_key_ptr,
        /* d_values_in */ value_ptr,
        /* d_values_out */ sorted_value_ptr,
        /* num_items */ num_row * num_col,
        /* num_segments */ num_row,
        /* d_begin_offsets */ segment_offsets_t,
        /* d_end_offsets */ segment_offsets_t + 1,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(T) * 8,
        /* stream */ cuda_stream);
    CudaCheck(err);
    CHECK_LE(infered_temp_storage_byte_size, temp_storage_bytes);

    err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        /* d_temp_storage */ temp_storage_ptr,
        /* temp_storage_bytes */ infered_temp_storage_byte_size,
        /* d_keys_in */ key_ptr,
        /* d_keys_out */ sorted_key_ptr,
        /* d_values_in */ value_ptr,
        /* d_values_out */ sorted_value_ptr,
        /* num_items */ num_row * num_col,
        /* num_segments */ num_row,
        /* d_begin_offsets */ segment_offsets_t,
        /* d_end_offsets */ segment_offsets_t + 1,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(T) * 8,
        /* stream */ cuda_stream);
    CudaCheck(err);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) template struct SortUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
