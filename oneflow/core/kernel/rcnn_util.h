#ifndef ONEFLOW_CORE_KERNEL_RCNN_UTIL_H_
#define ONEFLOW_CORE_KERNEL_RCNN_UTIL_H_

#include "oneflow/core/register/blob.h"

namespace oneflow {

template<typename T>
struct RcnnUtil {
  inline static float BBoxArea(const T* box_dptr);
  static float InterOverUnion(const T* box1_dptr, const T* box2_dptr);
  static void BboxOverlaps(const T* rois, const int32_t roi_num, const T* gt_boxes,
                           const int32_t gt_num, const int32_t gt_max_num, T* overlaps);
  static void OverlapRowArgMax7Max(const T* overlaps, const int32_t roi_mum, const int32_t gt_num,
                                   const int32_t gt_max_num, T* argmax_dptr, T* max_dptr);
  static void OverlapColArgMax7Max(const T* overlaps, const int32_t roi_mum, const int32_t gt_num,
                                   const int32_t gt_max_num, T* argmax_dptr, T* max_dptr);
  static void SampleChoice(T* rois_ptr, const int32_t num, T* sample_ptr, const int32_t sample_num);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RCNN_UTIL_H_
