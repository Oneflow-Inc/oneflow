#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
void FasterRcnnUtil<T>::BboxTransform(const T* bbox, const T* deltas, T* bbox_pred) {
  const float w = bbox[2] - bbox[0] + 1.0f;
  const float h = bbox[3] - bbox[1] + 1.0f;
  const float ctr_x = bbox[0] + 0.5f * w;
  const float ctr_y = bbox[1] + 0.5f * h;

  const float pred_ctr_x = deltas[0] * w + ctr_x;
  const float pred_ctr_y = deltas[1] * h + ctr_y;
  const float pred_w = std::exp(deltas[2]) * w;
  const float pred_h = std::exp(deltas[3]) * h;

  bbox_pred[0] = pred_ctr_x - 0.5f * pred_w;
  bbox_pred[1] = pred_ctr_y - 0.5f * pred_h;
  bbox_pred[2] = pred_ctr_x + 0.5f * pred_w - 1.f;
  bbox_pred[3] = pred_ctr_y + 0.5f * pred_h - 1.f;
}

template<typename T>
void FasterRcnnUtil<T>::BboxTransform(int64_t boxes_num, const T* bbox, const T* deltas,
                                      T* bbox_pred) {
  for (int64_t i = 0; i < boxes_num * 4; i += 4) {
    BboxTransform(bbox + i, deltas + i, bbox_pred + i);
  }
}

template<typename T>
void FasterRcnnUtil<T>::BboxTransformInv(int64_t boxes_num, const T* bbox, const T* target_bbox,
                                         T* deltas) {
  for (int64_t i = 0; i < boxes_num * 4; i += 4) {
    float b_w = bbox[i + 2] - bbox[i + 0] + 1.0f;
    float b_h = bbox[i + 3] - bbox[i + 1] + 1.0f;
    float b_ctr_x = bbox[i + 0] + 0.5f * b_w;
    float b_ctr_y = bbox[i + 1] + 0.5f * b_h;

    float t_w = target_bbox[i + 2] - target_bbox[i + 0] + 1.0f;
    float t_h = target_bbox[i + 3] - target_bbox[i + 1] + 1.0f;
    float t_ctr_x = target_bbox[i + 0] + 0.5f * t_w;
    float t_ctr_y = target_bbox[i + 1] + 0.5f * t_h;

    deltas[i + 0] = (t_ctr_x - b_ctr_x) / b_w;
    deltas[i + 1] = (t_ctr_y - b_ctr_y) / b_h;
    deltas[i + 2] = std::log(t_w / b_w);
    deltas[i + 3] = std::log(t_h / b_h);
  }
}

template<typename T>
void FasterRcnnUtil<T>::ClipBoxes(int64_t boxes_num, const int64_t image_height,
                                  const int64_t image_width, T* bbox) {
  for (int64_t i = 0; i < boxes_num * 4; i += 4) {
    bbox[i + 0] = std::max<T>(std::min<T>(bbox[i + 0], image_width), 0);
    bbox[i + 1] = std::max<T>(std::min<T>(bbox[i + 1], image_height), 0);
    bbox[i + 2] = std::max<T>(std::min<T>(bbox[i + 2], image_width), 0);
    bbox[i + 3] = std::max<T>(std::min<T>(bbox[i + 3], image_height), 0);
  }
}

template<typename T>
void FasterRcnnUtil<T>::SortByScore(const int64_t num, const T* score_ptr,
                                    int32_t* sorted_score_slice_ptr) {
  std::iota(sorted_score_slice_ptr, sorted_score_slice_ptr + num, 0);
  FOR_RANGE(int64_t, i, 0, num) { sorted_score_slice_ptr[i] = i; }
  std::sort(sorted_score_slice_ptr, sorted_score_slice_ptr + num,
            [&](int32_t lhs, int32_t rhs) { return score_ptr[lhs] > score_ptr[rhs]; });
}

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
  auto IsSuppressed = [&](int32_t index) -> bool {
    FOR_RANGE(int32_t, post_nms_slice_i, 0, keep_num) {
      const int32_t keep_index = post_nms_slice_ptr[post_nms_slice_i];
      const int32_t area0 = area_ptr[keep_index];
      const int32_t area1 = area_ptr[index];
      if (area0 >= area1 * nms_threshold && area1 >= area0 * nms_threshold) {
        const T* box0 = img_proposal_ptr + sorted_score_slice_ptr[keep_index] * 4;
        const T* box1 = img_proposal_ptr + sorted_score_slice_ptr[index] * 4;
        if (InterOverUnion(box0, area0, box1, area1) >= nms_threshold) { return true; }
      }
    }
    return false;
  };
  FOR_RANGE(int32_t, sorted_score_slice_i, 0, pre_nms_top_n) {
    if (IsSuppressed(sorted_score_slice_i)) { continue; }
    post_nms_slice_ptr[keep_num++] = sorted_score_slice_i;
    if (keep_num == post_nms_top_n) { break; }
  }
  FOR_RANGE(int32_t, i, 0, keep_num) {
    post_nms_slice_ptr[i] = sorted_score_slice_ptr[post_nms_slice_ptr[i]];
  }
  return keep_num;
}

#define INITIATE_FASTER_RCNN_UTIL(T, type_cpp) template struct FasterRcnnUtil<T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_FASTER_RCNN_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
