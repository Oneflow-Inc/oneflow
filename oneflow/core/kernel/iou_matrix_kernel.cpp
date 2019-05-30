#include "oneflow/core/kernel/iou_matrix_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void IoUMatrixKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* proposals_blob = BnInOp2Blob("proposals");
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  Blob* iou_matrix_blob = BnInOp2Blob("iou_matrix");
  Blob* iou_matrix_shape_blob = BnInOp2Blob("iou_matrix_shape");

  CHECK_EQ(proposals_blob->shape().At(1), 4);
  CHECK_EQ(gt_boxes_blob->shape().At(2), 4);
  const int32_t num_proposals = proposals_blob->shape().At(0);
  const int32_t num_imgs = gt_boxes_blob->shape().At(0);
  const T* proposals_ptr = proposals_blob->dptr<T>();
  int32_t num_gt_boxes = gt_boxes_blob->static_shape().At(1);
  FOR_RANGE(int32_t, img_idx, 0, num_imgs) {
    const T* gt_boxes_ptr = gt_boxes_blob->dptr<T>(img_idx);
    float* iou_matrix_ptr = iou_matrix_blob->mut_dptr<float>(img_idx);
    int32_t* iou_matrix_shape_ptr = iou_matrix_shape_blob->mut_dptr<int32_t>(img_idx);
    if (gt_boxes_blob->has_dim1_valid_num_field()) {
      num_gt_boxes = gt_boxes_blob->dim1_valid_num(img_idx);
    }
    IoUMatrixUtil<device_type, T>::ForwardSingleImage(
        ctx.device_ctx, proposals_ptr, num_proposals, gt_boxes_ptr, num_gt_boxes,
        gt_boxes_blob->static_shape().At(1), iou_matrix_ptr, iou_matrix_shape_ptr);
  }
}

template<typename T>
struct IoUMatrixUtil<DeviceType::kCPU, T> {
  static void ForwardSingleImage(DeviceCtx* ctx, const T* proposals_ptr,
                                 const int32_t num_proposals, const T* gt_boxes_ptr,
                                 const int32_t num_gt_boxes, const int32_t max_num_gt_boxes,
                                 float* iou_matrix_ptr, int32_t* iou_matrix_shape_ptr) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kIouMatrixConf, IoUMatrixKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
