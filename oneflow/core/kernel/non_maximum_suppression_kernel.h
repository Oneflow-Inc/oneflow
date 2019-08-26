#ifndef ONEFLOW_CORE_OPERATOR_NON_MAXIMUM_SUPPRESSION_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_NON_MAXIMUM_SUPPRESSION_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NonMaximumSuppressionKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonMaximumSuppressionKernel);
  NonMaximumSuppressionKernel() = default;
  ~NonMaximumSuppressionKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct NonMaximumSuppressionUtil {
  static void Forward(DeviceCtx* ctx, const size_t num_boxes, const float nms_iou_threshold,
                      const size_t num_keep, const T* boxes, int64_t* suppression, int8_t* keep);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NON_MAXIMUM_SUPPRESSION_KERNEL_OP_H_
