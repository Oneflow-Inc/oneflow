#ifndef ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_
#define ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T>
struct FasterRcnnUtil final {
  static int32_t Nms(const T* img_proposal_ptr, const int32_t* sorted_score_slice_ptr,
                     const int32_t pre_nms_top_n, const int32_t post_nms_top_n,
                     const float nms_threshold, int32_t* area_ptr, int32_t* post_nms_slice_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_
