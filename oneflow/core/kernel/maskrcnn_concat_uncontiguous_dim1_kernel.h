#ifndef ONEFLOW_CORE_OPERATOR_MASKRCNN_CONCAT_UNCONTIGUOUS_DIM1_KERNEL_H_
#define ONEFLOW_CORE_OPERATOR_MASKRCNN_CONCAT_UNCONTIGUOUS_DIM1_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaskrcnnConcatUncontiguousDim1Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaskrcnnConcatUncontiguousDim1Kernel);
  MaskrcnnConcatUncontiguousDim1Kernel() = default;
  ~MaskrcnnConcatUncontiguousDim1Kernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MASKRCNN_CONCAT_UNCONTIGUOUS_DIM1_KERNEL_H_
