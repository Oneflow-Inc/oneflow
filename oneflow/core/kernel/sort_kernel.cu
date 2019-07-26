#include "oneflow/core/kernel/sort_kernel.h"
#include "oneflow/core/kernel/gpu_radix_sort.cuh"

namespace oneflow {

template<typename T>
struct SortUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const T* key_ptr, const int32_t* value_ptr,
                      void* temp_storage_ptr, size_t temp_storage_bytes, int32_t num_row,
                      int32_t num_col, T* sorted_key_ptr, int32_t* sorted_value_ptr) {
    SortPairsDescending(key_ptr, value_ptr, num_row, num_col, temp_storage_ptr, temp_storage_bytes,
                        sorted_key_ptr, sorted_value_ptr, ctx->cuda_stream());
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) template struct SortUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
