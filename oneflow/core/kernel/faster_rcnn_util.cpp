#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
void FasterRcnnUtil<T>::BboxTransform(int64_t boxes_num, const T* bboxes, const T* deltas,
                                      T* pred_bboxes) {
  FOR_RANGE(int64_t, i, 0, boxes_num) {
    BBox<T>::MutCast(pred_bboxes)[i].Transform(BBox<T>::Cast(bboxes) + i,
                                               BBoxDelta<T>::Cast(deltas) + i);
  }
}

template<typename T>
void FasterRcnnUtil<T>::BboxTransformInv(int64_t boxes_num, const T* bboxes, const T* target_bboxes,
                                         T* deltas) {
  FOR_RANGE(int64_t, i, 0, boxes_num) {
    BBoxDelta<T>::MutCast(deltas)[i].TransformInverse(BBox<T>::Cast(bboxes) + i,
                                                      BBox<T>::Cast(target_bboxes) + i);
  }
}

template<typename T>
void FasterRcnnUtil<T>::ClipBoxes(int64_t boxes_num, const int64_t image_height,
                                  const int64_t image_width, T* bboxes) {
  BBox<T>* bbox_ptr = BBox<T>::MutCast(bboxes);
  FOR_RANGE(int64_t, i, 0, boxes_num) { bbox_ptr[i].Clip(image_height, image_width); }
}

template<typename T>
void FasterRcnnUtil<T>::SortByScore(const int64_t num, const T* score_ptr,
                                    int32_t* sorted_score_slice_ptr) {
  std::iota(sorted_score_slice_ptr, sorted_score_slice_ptr + num, 0);
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
    area_ptr[i] = (BBox<T>::Cast(img_proposal_ptr) + sorted_score_slice_ptr[i])->Area();
  }
  int32_t keep_num = 0;
  auto IsSuppressed = [&](int32_t index) -> bool {
    FOR_RANGE(int32_t, post_nms_slice_i, 0, keep_num) {
      const int32_t keep_index = post_nms_slice_ptr[post_nms_slice_i];
      const int32_t area0 = area_ptr[keep_index];
      const int32_t area1 = area_ptr[index];
      if (area0 >= area1 * nms_threshold && area1 >= area0 * nms_threshold) {
        const BBox<T>* box0 = BBox<T>::Cast(img_proposal_ptr) + sorted_score_slice_ptr[keep_index];
        const BBox<T>* box1 = BBox<T>::Cast(img_proposal_ptr) + sorted_score_slice_ptr[index];
        if (InterOverUnion(*box0, area0, *box1, area1) >= nms_threshold) { return true; }
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

template<typename T>
float FasterRcnnUtil<T>::InterOverUnion(const BBox<T>& box1, const int32_t area1,
                                        const BBox<T>& box2, const int32_t area2) {
  const int32_t iw = std::min(box1.x2(), box2.x2()) - std::max(box1.x1(), box2.x1()) + 1;
  if (iw <= 0) { return 0; }
  const int32_t ih = std::min(box1.y2(), box2.y2()) - std::max(box1.y1(), box2.y1()) + 1;
  if (ih <= 0) { return 0; }
  const float inter = iw * ih;
  return inter / (area1 + area2 - inter);
}

#define INITIATE_FASTER_RCNN_UTIL(T, type_cpp) template struct FasterRcnnUtil<T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_FASTER_RCNN_UTIL, FLOATING_DATA_TYPE_SEQ);

template<typename T>
void ScoredBBoxSlice<T>::Truncate(int64_t len) {
  CHECK_GE(len, 0);
  if (len < available_len_) { available_len_ = len; }
}

template<typename T>
void ScoredBBoxSlice<T>::TruncateByThreshold(float thresh) {
  int64_t keep_num = available_len_;
  FOR_RANGE(int64_t, i, 0, available_len_) {
    if (score_ptr_[index_slice_[i]] <= thresh) {
      keep_num = i;
      break;
    }
  }
  Truncate(keep_num);
}

template<typename T>
void ScoredBBoxSlice<T>::Concat(const ScoredBBoxSlice& other) {
  CHECK_LE(other.available_len(), len_ - available_len_);
  FOR_RANGE(int32_t, i, 0, other.available_len()) {
    index_slice_[available_len_ + i] = other.index_slice()[i];
  }
  available_len_ += other.available_len();
}

template<typename T>
void ScoredBBoxSlice<T>::DescSortByScore(bool init_index) {
  if (init_index) { std::iota(index_slice_, index_slice_ + available_len_, 0); }
  std::sort(index_slice_, index_slice_ + available_len_,
            [&](int32_t lhs, int32_t rhs) { return score_ptr_[lhs] > score_ptr_[rhs]; });
}

template<typename T>
void ScoredBBoxSlice<T>::NmsFrom(float nms_threshold, const ScoredBBoxSlice<T>& pre_nms_slice) {
  CHECK_NE(index_slice(), pre_nms_slice.index_slice());
  CHECK_EQ(bbox_ptr(), pre_nms_slice.bbox_ptr());
  CHECK_EQ(score_ptr(), pre_nms_slice.score_ptr());
  CHECK_LE(available_len(), pre_nms_slice.available_len());

  int32_t keep_num = 0;
  auto IsSuppressed = [&](int32_t pre_nms_slice_index) -> bool {
    const BBox<T>* cur_bbox = GetBBox(pre_nms_slice_index);
    FOR_RANGE(int32_t, post_nms_slice_i, 0, keep_num) {
      const BBox<T>* keep_bbox = GetBBox(index_slice_[post_nms_slice_i]);
      if (keep_bbox->InterOverUnion(cur_bbox) >= nms_threshold) { return true; }
    }
    return false;
  };
  FOR_RANGE(int32_t, pre_nms_slice_i, 0, pre_nms_slice.available_len()) {
    if (IsSuppressed(pre_nms_slice_i)) { continue; }
    index_slice_[keep_num++] = pre_nms_slice_i;
    if (keep_num == available_len_) { break; }
  }
  FOR_RANGE(int32_t, i, 0, keep_num) {
    index_slice_[i] = pre_nms_slice.index_slice()[index_slice_[i]];
  }
  Truncate(keep_num);
}

#define INITIATE_SCORED_BBOX_SLICE(T, type_cpp) template class ScoredBBoxSlice<T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_SCORED_BBOX_SLICE, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
