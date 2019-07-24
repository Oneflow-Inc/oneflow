#include "oneflow/core/kernel/sort_kernel.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

__global__ void SetOffset(int32_t* begin_offsets_ptr, int32_t* end_offsets_ptr, int32_t num_row,
                          int32_t num_col) {
  CUDA_1D_KERNEL_LOOP(i, num_row) {
    begin_offsets_ptr[i] = i * num_col;
    end_offsets_ptr[i] = (i + 1) * num_col;
  }
}

}  // namespace

template<typename T>
struct SortUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const T* key_ptr, const int32_t* value_ptr,
                      void* temp_storage_ptr, size_t temp_storage_bytes, int32_t num_row,
                      int32_t num_col, int32_t* begin_offsets_ptr, int32_t* end_offsets_ptr,
                      T* sorted_key_ptr, int32_t* sorted_value_ptr) {
    size_t infered_temp_storage_byte_size = -1;

    SetOffset<<<1, kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        begin_offsets_ptr, end_offsets_ptr, num_row, num_col);

    cudaStream_t cuda_stream = ctx->cuda_stream();

    // When d_temp_storage is NULL, no work is done and the required allocation size is returned in
    // temp_storage_bytes.
    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        /* d_temp_storage */ nullptr,
        /* temp_storage_bytes */ infered_temp_storage_byte_size,
        /* d_keys_in */ key_ptr,
        /* d_keys_out */ sorted_key_ptr,
        /* d_values_in */ value_ptr,
        /* d_values_out */ sorted_value_ptr,
        /* num_items */ num_row * num_col,
        /* num_segments */ num_row,
        /* d_begin_offsets */ begin_offsets_ptr,
        /* d_end_offsets */ end_offsets_ptr,
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
        /* d_begin_offsets */ begin_offsets_ptr,
        /* d_end_offsets */ end_offsets_ptr,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(T) * 8,
        /* stream */ cuda_stream);
    CudaCheck(err);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) template struct SortUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
