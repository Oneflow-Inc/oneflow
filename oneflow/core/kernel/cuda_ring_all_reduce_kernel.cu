#include "oneflow/core/kernel/cuda_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void AllReduce(CudaRingAllReduceArg<T> arg) {

}

}

template<typename T>
void CudaRingAllReduceKernelUtil<T>::AllReduce(DeviceCtx* ctx, CudaRingAllReduceArg<T> arg) {
  AllReduce<<<arg.num_rings, 256, 0, ctx->cuda_stream()>>>(arg);
}


#define INSTANTIATE_CUDA_RING_ALL_REDUCE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct CudaRingAllReduceKernelUtil<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CUDA_RING_ALL_REDUCE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
