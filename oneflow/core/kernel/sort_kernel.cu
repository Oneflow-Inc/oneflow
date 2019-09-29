#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/gpu_radix_sort.cuh"

namespace oneflow {

template<typename T>
void GpuSort(DeviceCtx* ctx, const T* in_ptr, int32_t instance_num, int32_t instance_size,
             std::string dir, void* temp_storage_ptr, size_t temp_storage_bytes, T* out_ptr) {
  if (dir == "ASCENDING") {
    SortKeysAscending(in_ptr, instance_num, instance_size, temp_storage_ptr, temp_storage_bytes,
                      out_ptr, ctx->cuda_stream());
  } else if (dir == "DESCENDING") {
    SortKeysDescending(in_ptr, instance_num, instance_size, temp_storage_ptr, temp_storage_bytes,
                       out_ptr, ctx->cuda_stream());
  } else {
    UNIMPLEMENTED();
  }
}

#define INSTANTIATE_GPU_ARG_SORT_TOP_K(T, type_proto)                                      \
  template void GpuSort<T>(DeviceCtx * ctx, const T* in_ptr, int32_t instance_num,         \
                           int32_t instance_size, std::string dir, void* temp_storage_ptr, \
                           size_t temp_storage_bytes, T* out_ptr);
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_ARG_SORT_TOP_K, ARITHMETIC_DATA_TYPE_SEQ)
#undef INSTANTIATE_GPU_ARG_SORT_TOP_K

}  // namespace oneflow
