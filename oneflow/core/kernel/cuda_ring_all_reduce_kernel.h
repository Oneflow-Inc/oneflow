#ifndef ONEFLOW_CORE_KERNEL_CUDA_RING_ALL_REDUCE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CUDA_RING_ALL_REDUCE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

constexpr int32_t CUDA_RING_ALL_REDUCE_MAX_NUM_LINK = 8;

template<typename T>
struct CudaRingAllReduceLinkParams {
  const T* recv;
  const T* src;
  T* send;
  T* dst;
  int64_t num_elem;
};

template<typename T>
struct CudaRingAllReduceParams {
  int32_t num_links;
  CudaRingAllReduceLinkParams<T> links[CUDA_RING_ALL_REDUCE_MAX_NUM_LINK];
};

template<typename T>
class CudaRingAllReduceKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaRingAllReduceKernel);
  CudaRingAllReduceKernel() = default;
  ~CudaRingAllReduceKernel() override = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
};

template<typename T>
struct CudaRingAllReduceKernelUtil {
  static void Send(DeviceCtx* ctx, CudaRingAllReduceParams<T> params);
  static void RecvReduceSend(DeviceCtx* ctx, CudaRingAllReduceParams<T> params);
  static void RecvReduceSendCopy(DeviceCtx* ctx, CudaRingAllReduceParams<T> params);
  static void RecvSendCopy(DeviceCtx* ctx, CudaRingAllReduceParams<T> params);
  static void RecvCopy(DeviceCtx* ctx, CudaRingAllReduceParams<T> params);
};

}  // namespace oneflow

#endif  // #define ONEFLOW_CORE_KERNEL_CUDA_RING_ALL_REDUCE_KERNEL_H_
