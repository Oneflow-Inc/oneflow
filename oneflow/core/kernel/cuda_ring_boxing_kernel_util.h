#ifndef ONEFLOW_CORE_KERNEL_CUDA_RING_BOXING_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_CUDA_RING_BOXING_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/reduce_method.pb.h"

namespace oneflow {

constexpr int32_t CUDA_RING_BOXING_MAX_NUM_LINK = 8;

template<typename T>
struct CudaRingBoxingLinkParams {
  const T* recv;
  const T* in;
  T* send;
  T* out;
  int64_t num_elem;
};

template<typename T>
struct CudaRingBoxingStepParams {
  int32_t num_links;
  CudaRingBoxingLinkParams<T> links[CUDA_RING_BOXING_MAX_NUM_LINK];
  bool recv;
  bool in;
  bool send;
  bool out;
};

template<ReduceMethod method, typename T>
struct CudaRingBoxingKernelUtil {
  static void LaunchGenericRingStep(DeviceCtx* ctx, CudaRingBoxingStepParams<T> params);
};

size_t GetCudaRingAllReducePackAlignSize();

}  // namespace oneflow

#endif  // #define ONEFLOW_CORE_KERNEL_CUDA_RING_BOXING_KERNEL_UTIL_H_
