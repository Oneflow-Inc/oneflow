#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace {

template<typename T>
inline int32_t BBoxArea(const T* box) {
  return (box[2] - box[0] + 1) * (box[3] - box[1] + 1);
}

template<typename T>
inline float InterOverUnion(const T* box0, const int32_t area0, const T* box1,
                            const int32_t area1) {
  const int32_t iw = std::min(box0[2], box1[2]) - std::max(box0[0], box1[0]) + 1;
  if (iw <= 0) { return 0; }
  const int32_t ih = std::min(box0[3], box1[3]) - std::max(box0[1], box1[1]) + 1;
  if (ih <= 0) { return 0; }
  const int32_t inter = iw * ih;
  return inter / (area0 + area1 - inter);
}

}  // namespace

template<typename T>
int32_t FasterRcnnUtil<T>::Nms(const T* img_proposal_ptr, const int32_t* sorted_score_slice_ptr,
                               const int32_t pre_nms_top_n, const int32_t post_nms_top_n,
                               const float nms_threshold, int32_t* area_ptr,
                               int32_t* post_nms_slice_ptr) {
  CHECK_NE(sorted_score_slice_ptr, post_nms_slice_ptr);
  FOR_RANGE(int32_t, i, 0, pre_nms_top_n) {
    area_ptr[i] = BBoxArea(img_proposal_ptr + sorted_score_slice_ptr[i] * 4);
  }
  int32_t keep_num = 0;
  auto IsSupressed = [&](int32_t index) -> bool {
    FOR_RANGE(int32_t, i, 0, keep_num) {
      const int32_t area0 = area_ptr[i];
      const int32_t area1 = area_ptr[index];
      if (area0 >= area1 * nms_threshold && area1 >= area0 * nms_threshold) {
        const T* box0 = img_proposal_ptr + sorted_score_slice_ptr[i] * 4;
        const T* box1 = img_proposal_ptr + sorted_score_slice_ptr[index] * 4;
        if (InterOverUnion(box0, area0, box1, area1) >= nms_threshold) { return true; }
      }
    }
    return false;
  };
  FOR_RANGE(int32_t, i, 0, pre_nms_top_n) {
    if (IsSupressed(i)) { continue; }
    post_nms_slice_ptr[keep_num++] = sorted_score_slice_ptr[i];
    if (keep_num == post_nms_top_n) { break; }
  }
  return keep_num;
}

#define INITIATE_FASTER_RCNN_UTIL(T, type_cpp) template struct FasterRcnnUtil<T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_FASTER_RCNN_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
