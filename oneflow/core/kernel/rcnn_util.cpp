#include "oneflow/core/kernel/rcnn_util.h"

namespace oneflow {

template<typename T>
float RcnnUtil<T>::ComputeArea(const T* box_dptr, int32_t offset) {
  return (box_dptr[offset + 2] - box_dptr[offset] + 1)
         * (box_dptr[offset + 3] - box_dptr[offset + 1] + 1);
}

template<typename T>
void RcnnUtil<T>::BboxOverlaps(DeviceCtx* ctx, const Blob* boxes, const Blob* gt_boxes,
                               Blob* overlaps) {
  const T* bbox_dptr = boxes->dptr<T>();
  const T* gt_bbox_dptr = gt_boxes->dptr<T>();
  T* overlaps_dptr = overlaps->mut_dptr<T>();
  FOR_RANGE(int32_t, i, 0, boxes->shape().At(0)) {
    FOR_RANGE(int32_t, j, 0, gt_boxes->shape().At(1)) {
      int32_t gt_id =
          i * gt_boxes->shape().At(1) * gt_boxes->shape().At(2) + j * gt_boxes->shape().At(2);
      float gt_area = ComputeArea(gt_bbox_dptr, gt_id);
      FOR_RANGE(int32_t, k, 0, boxes->shape().At(1)) {
        int32_t box_id = i * boxes->shape().At(1) * boxes->shape().At(2) + k * boxes->shape().At(2);
        int32_t iw = std::min(bbox_dptr[box_id + 2], gt_bbox_dptr[gt_id + 2])
                     - std::max(bbox_dptr[box_id], gt_bbox_dptr[gt_id]) + 1;
        if (iw > 0) {
          int32_t ih = std::min(bbox_dptr[box_id + 3], gt_bbox_dptr[gt_id + 3])
                       - std::max(bbox_dptr[box_id + 1], gt_bbox_dptr[gt_id + 1]) + 1;
          if (ih > 0) {
            float ua = ComputeArea(bbox_dptr, box_id) + gt_area - iw * ih;
            int32_t overlap_id = i * boxes->shape().At(1) * gt_boxes->shape().At(1)
                                 + k * gt_boxes->shape().At(1) + j;
            overlaps_dptr[overlap_id] = iw * ih / ua;
          }
        }
      }
    }
  }
}

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto) template struct RcnnUtil<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
