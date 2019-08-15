#ifndef ONEFLOW_CORE_OPERATOR_MASKRCNN_POSITIVE_NEGATIVE_SAMPLE_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_MASKRCNN_POSITIVE_NEGATIVE_SAMPLE_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaskrcnnPositiveNegativeSampleKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaskrcnnPositiveNegativeSampleKernel);
  MaskrcnnPositiveNegativeSampleKernel() = default;
  ~MaskrcnnPositiveNegativeSampleKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MASKRCNN_POSITIVE_NEGATIVE_SAMPLE_KERNEL_OP_H_
