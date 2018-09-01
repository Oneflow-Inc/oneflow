#ifndef ONEFLOW_CORE_KERNEL_HINGE_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_HINGE_LOSS_KERNEL_H_

#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
class HingeLossKernel final : public LossKernel<device_type, PredType, LabelType> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HingeLossKernel);
  HingeLossKernel() = default;
  ~HingeLossKernel() = default;

 private:
  void VirtualLossForwardDataContent(const KernelCtx&,
                                     std::function<Blob*(const std::string&)>) const override;
  const LossKernelConf& GetLossKernelConf(const KernelConf& kernel_conf) const override;
};

template<DeviceType device_type, typename PredType, typename LabelType>
struct HingeLossKernelUtil {
  static void Forward(DeviceCtx* ctx, const int64_t piece_size, const int64_t pre_dim,
                      const PredType* pred, const LabelType* label, const OperatorConf& op_conf,
                      PredType* tmp_diff, PredType* tmp, PredType* tmp_storage, PredType* loss);
  static void Backward(DeviceCtx* ctx, const int64_t piece_size, const int64_t pre_dim,
                       const PredType* tmp_diff, const LabelType* label,
                       const OperatorConf& op_conf, PredType* pred_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_HINGE_LOSS_KERNEL_H_
