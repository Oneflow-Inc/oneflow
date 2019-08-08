#ifndef ONEFLOW_CORE_KERNEL_CUDA_RING_BOXING_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_CUDA_RING_BOXING_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/reduce_method.pb.h"

namespace oneflow {

#ifdef WITH_CUDA

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

template<ReduceMethod method>
struct CudaRingBoxingKernelUtil<method, float16> {
  static void LaunchGenericRingStep(DeviceCtx* ctx, CudaRingBoxingStepParams<float16> params);
};

size_t GetCudaRingBoxingPackCoalesceRegionSize();

#endif

}  // namespace oneflow

#endif  // #define ONEFLOW_CORE_KERNEL_CUDA_RING_BOXING_KERNEL_UTIL_H_
