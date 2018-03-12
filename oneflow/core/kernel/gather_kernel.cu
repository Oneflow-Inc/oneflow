#include "oneflow/core/kernel/gather_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GatherGpu(const int64_t n, const int32_t hidden_dim,
                          const int32_t col_id, const T* src_dptr,
                          const int32_t* col_num_ptr, T* dst_dptr) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (col_num_ptr[i / hidden_dim] == col_id) { dst_dptr[i] = src_dptr[i]; }
  }
}

}  // namespace

template<typename T>
class GatherKernelUtil<DeviceType::kGPU, T> {
 public:
  static void Gather(DeviceCtx* ctx, const int64_t n, const int32_t hidden_dim,
                     const int32_t col_id, const T* src_dptr,
                     const int32_t* col_num_ptr, T* dst_dptr) {
    GatherGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                ctx->cuda_stream()>>>(n, hidden_dim, col_id, src_dptr,
                                      col_num_ptr, dst_dptr);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class GatherKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
