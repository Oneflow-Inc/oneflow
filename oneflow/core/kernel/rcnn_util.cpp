#include "oneflow/core/kernel/rcnn_util.h"

namespace oneflow {

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
      float gt_area = (gt_bbox_dptr[gt_id + 2] - gt_bbox_dptr[gt_id] + 1)
                      * (gt_bbox_dptr[gt_id + 3] - gt_bbox_dptr[gt_id + 1] + 1);
      FOR_RANGE(int32_t, k, 0, boxes->shape().At(1)) {
        int32_t box_id = i * boxes->shape().At(1) * boxes->shape().At(2) + k * boxes->shape().At(2);
        int32_t iw = std::min(bbox_dptr[box_id + 2], gt_bbox_dptr[gt_id + 2])
                     - std::max(bbox_dptr[box_id], gt_bbox_dptr[gt_id]) + 1;
        if (iw > 0) {
          int32_t ih = std::min(bbox_dptr[box_id + 3], gt_bbox_dptr[gt_id + 3])
                       - std::max(bbox_dptr[box_id + 1], gt_bbox_dptr[gt_id + 1]) + 1;
          if (ih > 0) {
            float ua = (bbox_dptr[box_id + 2] - bbox_dptr[box_id] + 1)
                           * (bbox_dptr[box_id + 3] - bbox_dptr[box_id + 1] + 1)
                       + gt_area - iw * ih;
            int32_t overlap_id = i * boxes->shape().At(1) * gt_boxes->shape().At(1)
                                 + k * gt_boxes->shape().At(1) + j;
            overlaps_dptr[overlap_id] = iw * ih / ua;
          }
        }
      }
    }
  }
}

}  // namespace oneflow
