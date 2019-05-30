#include "oneflow/core/kernel/iou_matrix_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GpuForwardSingleImage(const T* proposals_ptr, const int32_t num_proposals,
                                      const T* gt_boxes_ptr, const int32_t num_gt_boxes,
                                      const int32_t max_num_gt_boxes, float* iou_matrix_ptr,
                                      int32_t* iou_matrix_shape_ptr) {
  const T TO_REMOVE = 1.0;
  CUDA_1D_KERNEL_LOOP(i, num_proposals) {
#pragma unroll
    FOR_RANGE(int32_t, j, 0, num_gt_boxes) {
      const T proposal_x1 = proposals_ptr[i * 4];
      const T proposal_y1 = proposals_ptr[i * 4 + 1];
      const T proposal_x2 = proposals_ptr[i * 4 + 2];
      const T proposal_y2 = proposals_ptr[i * 4 + 3];
      const T gt_box_x1 = gt_boxes_ptr[j * 4];
      const T gt_box_y1 = gt_boxes_ptr[j * 4 + 1];
      const T gt_box_x2 = gt_boxes_ptr[j * 4 + 2];
      const T gt_box_y2 = gt_boxes_ptr[j * 4 + 3];
      const T iw = min(proposal_x2, gt_box_x2) - max(proposal_x1, gt_box_x1) + TO_REMOVE;
      const T ih = min(proposal_y2, gt_box_y2) - max(proposal_y1, gt_box_y1) + TO_REMOVE;
      const T area_inner = iw * ih;
      const T area_proposal =
          (proposal_y2 - proposal_y1 + TO_REMOVE) * (proposal_x2 - proposal_x1 + TO_REMOVE);
      const T area_gt_box =
          (gt_box_y2 - gt_box_y1 + TO_REMOVE) * (gt_box_x2 - gt_box_x1 + TO_REMOVE);
      iou_matrix_ptr[i * max_num_gt_boxes + j] =
          area_inner / (area_proposal + area_gt_box - area_inner);
    }
  }
  iou_matrix_shape_ptr[0] = num_proposals;
  iou_matrix_shape_ptr[1] = num_gt_boxes;
}

}  // namespace

template<typename T>
struct IoUMatrixUtil<DeviceType::kGPU, T> {
  static void ForwardSingleImage(DeviceCtx* ctx, const T* proposals_ptr,
                                 const int32_t num_proposals, const T* gt_boxes_ptr,
                                 const int32_t num_gt_boxes, const int32_t max_num_gt_boxes,
                                 float* iou_matrix_ptr, int32_t* iou_matrix_shape_ptr) {
    GpuForwardSingleImage<<<std::min(num_proposals, kCudaMaxBlocksNum), kCudaThreadsNumPerBlock, 0,
                            ctx->cuda_stream()>>>(proposals_ptr, num_proposals, gt_boxes_ptr,
                                                  num_gt_boxes, max_num_gt_boxes, iou_matrix_ptr,
                                                  iou_matrix_shape_ptr);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) template struct IoUMatrixUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
