#ifndef ONEFLOW_CORE_OPERATOR_BOX_DECODE_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_BOX_DECODE_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BoxDecodeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxDecodeKernel);
  BoxDecodeKernel() = default;
  ~BoxDecodeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct BoxDecodeUtil {
  static void Forward(DeviceCtx* ctx, const int32_t num_boxes, const T* ref_boxes_ptr,
                      const T* boxes_delta_ptr, const float weight_x, const float weight_y,
                      const float weight_w, const float weight_h, T* boxes_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BOX_DECODE_KERNEL_OP_H_
