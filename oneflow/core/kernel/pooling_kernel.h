#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PoolingKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel();
  ~PoolingKernel();

  void InitFromOpProto(const OperatorProto& op_proto) override;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;

#ifdef USE_CUDNN
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnPoolingMode_t pooling_mode_;
  cudnnPoolingDescriptor_t pooling_desc_;
#endif
};

template<DeviceType device_type, typename T>
class PoolingKernelUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelUtil);
  PoolingKernelUtil() = delete;

  static void PoolingForward(const KernelCtx&, const Blob*, Blob*, Blob*,
                             const PoolingOpConf&);

  static void PoolingBackward(const KernelCtx&, const Blob*, const Blob*, Blob*,
                              const PoolingOpConf&);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
