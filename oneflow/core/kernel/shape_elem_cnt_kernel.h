#ifndef ONEFLOW_CORE_KERNEL_SHAPE_ELEM_CNT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SHAPE_ELEM_CNT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ShapeElemCntKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ShapeElemCntKernel);
  ShapeElemCntKernel() = default;
  ~ShapeElemCntKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  int32_t GetShapePartialElemCnt(const ShapeView& shape) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SHAPE_ELEM_CNT_KERNEL_H_
