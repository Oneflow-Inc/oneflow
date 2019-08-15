#ifndef ONEFLOW_CORE_OPERATOR_CALC_IOU_MATRIX_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_CALC_IOU_MATRIX_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class CalcIoUMatrixKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CalcIoUMatrixKernel);
  CalcIoUMatrixKernel() = default;
  ~CalcIoUMatrixKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardInstanceShape(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct CalcIoUMatrixUtil {
  static void Forward(DeviceCtx* ctx, const T* boxes1_ptr, const int32_t num_boxes1,
                      const T* boxes2_ptr, const int32_t num_boxes2, float* iou_matrix_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CALC_IOU_MATRIX_KERNEL_OP_H_
