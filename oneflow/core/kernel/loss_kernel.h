#ifndef ONEFLOW_CORE_KERNEL_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOSS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType>
class LossKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossKernel);
  LossKernel() = default;
  virtual ~LossKernel() = default;

 protected:
  virtual void VirtualLossForwardDataContent(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual const LossKernelConf& GetLossKernelConf(const KernelConf& kernel_conf) const = 0;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
struct LossKernelUtil {
  static void ComputeReductionCoefficient(DeviceCtx* ctx, int64_t data_num, int64_t weight_length,
                                          const T* weight, T* reduction, ScalarReductionType type);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOSS_KERNEL_H_
