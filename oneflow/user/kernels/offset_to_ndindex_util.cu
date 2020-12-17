#ifdef WITH_CUDA
#include <cub/cub.cuh>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/offset_to_ndindex_util.h"

namespace oneflow {

namespace user_op {

__device__ void checkOffsetGPU(int32_t offset, int32_t dims_elem_cnt) {
  if(offset > dims_elem_cnt){
    __trap();
  }
}

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