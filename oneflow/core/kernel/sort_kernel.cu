#include "oneflow/core/kernel/sort_kernel.h"
#include "oneflow/core/kernel/radix_sort_util.cuh"
#include <cub/cub.cuh>

namespace oneflow {

template<typename T>
struct SortUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const T* key_ptr, const int32_t* value_ptr,
                      void* temp_storage_ptr, size_t temp_storage_bytes, int32_t num_row,
                      int32_t num_col, T* sorted_key_ptr, int32_t* sorted_value_ptr) {
    cub::CountingInputIterator<int32_t> counting_iter(0);
    cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
        segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

    cudaStream_t cuda_stream = ctx->cuda_stream();

    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        /* d_temp_storage */ temp_storage_ptr,
        /* temp_storage_bytes */ temp_storage_bytes,
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
