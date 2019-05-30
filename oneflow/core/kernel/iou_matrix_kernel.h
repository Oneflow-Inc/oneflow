#ifndef ONEFLOW_CORE_OPERATOR_IOU_MATRIX_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_IOU_MATRIX_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class IoUMatrixKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IoUMatrixKernel);
  IoUMatrixKernel() = default;
  ~IoUMatrixKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct IoUMatrixUtil {
  static void ForwardSingleImage(DeviceCtx* ctx, const T* proposals_ptr,
                                 const int32_t num_proposals, const T* gt_boxes_ptr,
                                 const int32_t num_gt_boxes, const int32_t max_num_gt_boxes,
                                 float* iou_matrix_ptr, int32_t* iou_matrix_shape_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_IOU_MATRIX_KERNEL_OP_H_
