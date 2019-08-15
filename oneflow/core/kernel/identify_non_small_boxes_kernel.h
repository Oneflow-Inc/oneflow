#ifndef ONEFLOW_CORE_OPERATOR_IDENTIFY_NON_SMALL_BOXES_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_IDENTIFY_NON_SMALL_BOXES_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class IdentifyNonSmallBoxesKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentifyNonSmallBoxesKernel);
  IdentifyNonSmallBoxesKernel() = default;
  ~IdentifyNonSmallBoxesKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct IdentifyNonSmallBoxesUtil {
  static void Forward(DeviceCtx* ctx, const T* in_ptr, const int32_t num_boxes,
                      const float min_size, int8_t* out_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_IDENTIFY_NON_SMALL_BOXES_KERNEL_OP_H_
