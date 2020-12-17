#ifdef WITH_CUDA
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/offset_to_ndindex_util.h"

namespace oneflow {

namespace user_op {

template<typename T>
__global__ void OffsetToIndexForwardGpuKernel(int32_t in_num, int32_t ndim, const T* index, const T* dims_tensor, T* out) {
  DoOffsetToIndex<T>(in_num, ndim, index, dims_tensor, out);
  // check_shape();
  
}

template<typename T>
struct OffsetToNdIndexFunctor<DeviceType::kGPU, T> final {
    void operator()(DeviceCtx* ctx, int32_t in_num,
        int32_t ndim, const T* index, const T* dims_tensor, T* out) {
    RUN_CUDA_KERNEL((OffsetToIndexForwardGpuKernel<T>), ctx, in_num, in_num, ndim, index, dims_tensor, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_OFFSET_TO_NDINDEX_FUNCTOR, (DeviceType::kGPU),
OFFSET_TO_NDINDEX_DATA_TYPE_SEQ);
}  // namespace user_op
}  // namespace oneflow

#endif  // End WITH_CUDA