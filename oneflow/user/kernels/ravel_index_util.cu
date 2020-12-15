#ifdef WITH_CUDA
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/ravel_index_util.h"


namespace oneflow {

namespace user_op {

template<typename T>
__global__ void RavelIndexForwardGpuKernel(int32_t in_num, 
  int32_t ndim, const T* index, const T* dims_tensor, T* out) {
  printf("RUN CUDA KERNEL");
  DoIndexToOffset<T>(in_num, ndim, index, dims_tensor, out);
}

template<typename T>
struct RavelIndexFunctor<DeviceType::kGPU, T> final {
    void operator()(DeviceCtx* ctx, int32_t in_num,
        int32_t ndim, const T* index, const T* dims_tensor, T* out) {
    printf("Enter Cuda operator");
    RUN_CUDA_KERNEL((RavelIndexForwardGpuKernel<T>), ctx, in_num, in_num, ndim, index, dims_tensor, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_RAVEL_INDEX_FUNCTOR, (DeviceType::kGPU),
                                 RAVEL_INDEX_DATA_TYPE_SEQ);
}  // namespace user_op
}  // namespace oneflow

#endif  // End WITH_CUDA