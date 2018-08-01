#ifndef ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_
#define ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T>
struct FasterRcnnUtil final {
  inline static int32_t BBoxArea(const T* box) {
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1);
  }

  inline static float InterOverUnion(const T* box0, const int32_t area0, const T* box1,
                                     const int32_t area1) {
    const int32_t iw = std::min(box0[2], box1[2]) - std::max(box0[0], box1[0]) + 1;
    if (iw <= 0) { return 0; }
    const int32_t ih = std::min(box0[3], box1[3]) - std::max(box0[1], box1[1]) + 1;
    if (ih <= 0) { return 0; }
    const float inter = iw * ih;
    return inter / (area0 + area1 - inter);
  }

  static void BboxTransform(int64_t boxes_num, const T* bbox, const T* deltas, T* bbox_pred);
  static void BboxTransformInv(int64_t boxes_num, const T* bbox, const T* target_bbox, T* deltas);
  static void ClipBoxes(int64_t boxes_num, const int64_t image_height, const int64_t image_width,
                        T* bbox);
  static int32_t Nms(const T* img_proposal_ptr, const int32_t* sorted_score_slice_ptr,
                     const int32_t pre_nms_top_n, const int32_t post_nms_top_n,
                     const float nms_threshold, int32_t* area_ptr, int32_t* post_nms_slice_ptr);
  static void SortByScore(const int64_t num, const T* score_ptr, int32_t* sorted_score_slice_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_
