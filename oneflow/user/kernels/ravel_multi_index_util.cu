#ifdef WITH_CUDA
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/ravel_multi_index_util.h"


namespace oneflow {

namespace user_op {

template<typename T>
__global__ void RavelMultiIndexForwardGpuKernel(int32_t n, int32_t in_num, 
    const RavelMultiIndexHelper<T> helper, const T** in_dptrs, T* out) {
  printf("RUN CUDA KERNEL");
  DoIndexToOffset<T>(n, in_num, helper, in_dptrs, out);
}

template<typename T>
struct RavelMultiIndexFunctor<DeviceType::kGPU, T> final {
  void operator()(DeviceCtx* ctx, user_op::KernelComputeContext* kernel_ctx,  int32_t n, int32_t in_num,
                  int32_t ndim, const Tensor* dims_tensor, T* out) {
    const T* dims = dims_tensor->dptr<T>();
    std::cout<<"Helper Ndim is: "<<ndim<<std::endl;
    RavelMultiIndexHelper<T> helper(dims, ndim);

    printf("Enter Cuda operator");

    const T* in_dptrs[6]; // The max input num is 6
    printf("ALlocate dptrs");
    // TODO: Add a check to promise the input num is less than the max legal input num
    for (int32_t i = 0; i < in_num; ++i) {
      printf("Current Loop idx is %d", i);
      in_dptrs[i] = kernel_ctx->Tensor4ArgNameAndIndex("multi_index", i)->dptr<T>();
      std::cout<<"In dptrs[i] is: "<<in_dptrs[i]<<std::endl;
    }     
    printf("RUN FUNCTOR");
    RUN_CUDA_KERNEL((RavelMultiIndexForwardGpuKernel<T>), ctx, in_num, n, in_num, helper, in_dptrs, out);
  }
};


OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_RAVEL_MULTI_INDEX_FUNCTOR, (DeviceType::kGPU),
                                 RAVEL_MULTI_INDEX_DATA_TYPE_SEQ);
}  // namespace user_op
}  // namespace oneflow

#endif  // End WITH_CUDA