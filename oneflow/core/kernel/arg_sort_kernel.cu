#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/gpu_radix_sort.cuh"

namespace oneflow {

namespace {

__global__ void RadixSortTopKInitializeKernel(int32_t* indices_ptr, int32_t instance_size) {
  for (int32_t i = threadIdx.x; i < instance_size; i += blockDim.x) {
    indices_ptr[blockIdx.x * instance_size + i] = i;
  }
}

}  // namespace

template<typename T>
void GpuArgSort(DeviceCtx* ctx, const T* in_ptr, int32_t* indices_ptr, int32_t instance_num,
                int32_t instance_size, std::string dir, void* temp_storage_ptr,
                size_t temp_storage_bytes, T* sorted_in_ptr, int32_t* out_ptr) {
  int32_t num_thread =
      instance_size <= kCudaThreadsNumPerBlock ? instance_size : kCudaThreadsNumPerBlock;
  RadixSortTopKInitializeKernel<<<instance_num, num_thread, 0, ctx->cuda_stream()>>>(indices_ptr,
                                                                                     instance_size);
  if (dir == "ASCENDING") {
    SortPairsAscending(in_ptr, indices_ptr, instance_num, instance_size, temp_storage_ptr,
                       temp_storage_bytes, sorted_in_ptr, out_ptr, ctx->cuda_stream());
  } else if (dir == "DESCENDING") {
    SortPairsDescending(in_ptr, indices_ptr, instance_num, instance_size, temp_storage_ptr,
                        temp_storage_bytes, sorted_in_ptr, out_ptr, ctx->cuda_stream());
  } else {
    UNIMPLEMENTED();
  }
}

#define INSTANTIATE_GPU_ARG_SORT_TOP_K(T, type_proto)                                              \
  template void GpuArgSort<T>(DeviceCtx * ctx, const T* in_ptr, int32_t* indices_ptr,              \
                              int32_t instance_num, int32_t instance_size, std::string dir,        \
                              void* temp_storage_ptr, size_t temp_storage_bytes, T* sorted_in_ptr, \
                              int32_t* out_ptr);
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_ARG_SORT_TOP_K, ARITHMETIC_DATA_TYPE_SEQ)
#undef INSTANTIATE_GPU_ARG_SORT_TOP_K

}  // namespace oneflow
