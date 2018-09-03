#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
void FasterRcnnUtil<T>::GenerateAnchors(const AnchorGeneratorConf& conf, Blob* anchors_blob) {
  // anchors_blob shape (h, w, a, 4)
  const int32_t height = anchors_blob->shape().At(0);
  const int32_t width = anchors_blob->shape().At(1);
  const int32_t scales_size = conf.anchor_scales_size();
  const int32_t ratios_size = conf.aspect_ratios_size();
  const int32_t fm_stride = conf.feature_map_stride();
  const int32_t num_anchors = scales_size * ratios_size;
  CHECK_EQ(num_anchors, anchors_blob->shape().At(2));

  const float base_ctr = 0.5 * (fm_stride - 1);
  std::vector<T> base_anchors(num_anchors * 4);
  BBox<T>* base_anchor_bbox = BBox<T>::MutCast(base_anchors.data());
  FOR_RANGE(int32_t, i, 0, ratios_size) {
    const int32_t wr = std::round(std::sqrt(fm_stride * fm_stride / conf.aspect_ratios(i)));
    const int32_t hr = std::round(wr * conf.aspect_ratios(i));
    FOR_RANGE(int32_t, j, 0, scales_size) {
      const float scale = conf.anchor_scales(j) / fm_stride;
      const int32_t ws = wr * scale;
      const int32_t hs = hr * scale;
      BBox<T>* cur_anchor_bbox = base_anchor_bbox + i * scales_size + j;
      cur_anchor_bbox->set_x1(base_ctr - 0.5 * (ws - 1));
      cur_anchor_bbox->set_y1(base_ctr - 0.5 * (hs - 1));
      cur_anchor_bbox->set_x2(base_ctr + 0.5 * (ws - 1));
      cur_anchor_bbox->set_y2(base_ctr + 0.5 * (hs - 1));
    }
  }

  const BBox<T>* const_base_anchor_bbox = BBox<T>::Cast(base_anchors.data());
  FOR_RANGE(int32_t, h, 0, height) {
    FOR_RANGE(int32_t, w, 0, width) {
      BBox<T>* anchor_bbox = BBox<T>::MutCast(anchors_blob->mut_dptr<T>(h, w));
      FOR_RANGE(int32_t, i, 0, num_anchors) {
        anchor_bbox[i].set_x1(const_base_anchor_bbox[i].x1() + w * fm_stride);
        anchor_bbox[i].set_y1(const_base_anchor_bbox[i].y1() + h * fm_stride);
        anchor_bbox[i].set_x2(const_base_anchor_bbox[i].x2() + w * fm_stride);
        anchor_bbox[i].set_y2(const_base_anchor_bbox[i].y2() + h * fm_stride);
      }
    }
  }
}

template<typename T>
void FasterRcnnUtil<T>::BboxTransform(int64_t boxes_num, const T* bboxes, const T* deltas,
                                      const BBoxRegressionWeights& bbox_reg_ws, T* pred_bboxes) {
  FOR_RANGE(int64_t, i, 0, boxes_num) {
    BBox<T>::MutCast(pred_bboxes)[i].Transform(BBox<T>::Cast(bboxes) + i,
                                               BBoxDelta<T>::Cast(deltas) + i, bbox_reg_ws);
  }
}

template<typename T>
void FasterRcnnUtil<T>::BboxTransformInv(int64_t boxes_num, const T* bboxes, const T* target_bboxes,
                                         const BBoxRegressionWeights& bbox_reg_ws, T* deltas) {
  FOR_RANGE(int64_t, i, 0, boxes_num) {
    BBoxDelta<T>::MutCast(deltas)[i].TransformInverse(
        BBox<T>::Cast(bboxes) + i, BBox<T>::Cast(target_bboxes) + i, bbox_reg_ws);
  }
}

template<typename T>
void FasterRcnnUtil<T>::ClipBoxes(int64_t boxes_num, const int64_t image_height,
                                  const int64_t image_width, T* bboxes) {
  BBox<T>* bbox_ptr = BBox<T>::MutCast(bboxes);
  FOR_RANGE(int64_t, i, 0, boxes_num) { bbox_ptr[i].Clip(image_height, image_width); }
}

template<typename T>
size_t FasterRcnnUtil<T>::ConvertGtBoxesToAbsoluteCoord(const FloatList16* gt_boxes,
                                                        const size_t image_height,
                                                        const size_t image_width,
                                                        T* converted_gt_boxes) {
  int32_t boxes_num = gt_boxes->value().value_size() / 4;
  const BBox<float>* gt_bbox = BBox<float>::Cast(gt_boxes->value().value().data());
  BBox<T>* converted_gt_bbox = BBox<T>::MutCast(converted_gt_boxes);
  FOR_RANGE(int32_t, i, 0, boxes_num) {
    converted_gt_bbox[i].set_x1(gt_bbox[i].x1() * image_width);
    converted_gt_bbox[i].set_y1(gt_bbox[i].y1() * image_height);
    converted_gt_bbox[i].set_x2(gt_bbox[i].x2() * image_width - 1);
    converted_gt_bbox[i].set_y2(gt_bbox[i].y2() * image_height - 1);
  }
  return boxes_num;
}

template<typename T>
void FasterRcnnUtil<T>::ForEachOverlapBetweenBoxesAndGtBoxes(
    const BoxesSlice<T>& boxes_slice, const BoxesSlice<T>& gt_boxes_slice,
    const std::function<void(int32_t, int32_t, float)>& Handler) {
  FOR_RANGE(int32_t, i, 0, gt_boxes_slice.size()) {
    FOR_RANGE(int32_t, j, 0, boxes_slice.size()) {
      float overlap = boxes_slice.GetBBox(j)->InterOverUnion(gt_boxes_slice.GetBBox(i));
      Handler(boxes_slice.GetIndex(j), gt_boxes_slice.GetIndex(i), overlap);
    }
  }
}

template<typename T>
void FasterRcnnUtil<T>::ForEachOverlapBetweenBoxesAndGtBoxes(
    const BoxesSlice<T>& boxes_slice, const GtBoxes<FloatList16>& gt_boxes_slice,
    const std::function<void(int32_t, int32_t, float)>& Handler) {
  FOR_RANGE(int32_t, i, 0, gt_boxes_slice.size()) {
    FOR_RANGE(int32_t, j, 0, boxes_slice.size()) {
      float overlap = boxes_slice.GetBBox(j)->InterOverUnion(gt_boxes_slice.GetBBox<float>(i));
      Handler(boxes_slice.GetIndex(j), i, overlap);
    }
  }
}

#define INITIATE_FASTER_RCNN_UTIL(T, type_cpp) template struct FasterRcnnUtil<T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_FASTER_RCNN_UTIL, FLOATING_DATA_TYPE_SEQ);

template<typename T>
void ScoredBBoxSlice<T>::Truncate(int32_t len) {
  CHECK_GE(len, 0);
  if (len < available_len_) { available_len_ = len; }
}

template<typename T>
void ScoredBBoxSlice<T>::TruncateByThreshold(const float thresh) {
  Truncate(FindByThreshold(thresh));
}

// Find first index which score less than threshold.
template<typename T>
int32_t ScoredBBoxSlice<T>::FindByThreshold(const float thresh) {
  FOR_RANGE(int32_t, i, 0, available_len_) {
    if (score_ptr_[index_slice_[i]] < thresh) { return i; }
  }
  return available_len_;
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
void ScoredBBoxSlice<T>::Sort(
    const std::function<bool(const T, const T, const BBox<T>&, const BBox<T>&)>& Compare) {
  std::sort(index_slice_, index_slice_ + available_len_, [&](int32_t l_idx, int32_t r_idx) {
    const BBox<T>* bbox = BBox<T>::Cast(bbox_ptr_);
    return Compare(score_ptr_[l_idx], score_ptr_[r_idx], bbox[l_idx], bbox[r_idx]);
  });
}

template<typename T>
void ScoredBBoxSlice<T>::Filter(const std::function<bool(const T, const BBox<T>*)>& IsFiltered) {
  int32_t keep_num = 0;
  FOR_RANGE(int32_t, i, 0, available_len_) {
    if (!IsFiltered(GetScore(i), GetBBox(i))) {
      // keep_num <= i so index_slice_ never be written before read
      index_slice_[keep_num++] = index_slice_[i];
    }
  }
  available_len_ = keep_num;
}

template<typename T>
ScoredBBoxSlice<T> ScoredBBoxSlice<T>::Slice(const int32_t begin, const int32_t end) {
  CHECK_GT(end, begin);
  CHECK_GE(begin, 0);
  CHECK_LE(end, available_len_);
  return ScoredBBoxSlice(end - begin, bbox_ptr_, score_ptr_, index_slice_ + begin);
}

template<typename T>
void ScoredBBoxSlice<T>::Shuffle() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(index_slice_, index_slice_ + available_len_, gen);
}

template<typename T>
void ScoredBBoxSlice<T>::NmsFrom(float nms_threshold, const ScoredBBoxSlice<T>& pre_nms_slice) {
  CHECK_NE(index_slice(), pre_nms_slice.index_slice());
  CHECK_EQ(bbox_ptr(), pre_nms_slice.bbox_ptr());
  CHECK_EQ(score_ptr(), pre_nms_slice.score_ptr());

  int32_t keep_num = 0;
  auto IsSuppressed = [&](int32_t pre_nms_slice_index) -> bool {
    const BBox<T>* cur_bbox = pre_nms_slice.GetBBox(pre_nms_slice_index);
    FOR_RANGE(int32_t, post_nms_slice_i, 0, keep_num) {
      const BBox<T>* keep_bbox = GetBBox(post_nms_slice_i);
      if (keep_bbox->InterOverUnion(cur_bbox) >= nms_threshold) { return true; }
    }
    return false;
  };
  FOR_RANGE(int32_t, pre_nms_slice_i, 0, pre_nms_slice.available_len()) {
    if (IsSuppressed(pre_nms_slice_i)) { continue; }
    index_slice_[keep_num++] = pre_nms_slice.GetSlice(pre_nms_slice_i);
    if (keep_num == available_len_) { break; }
  }
  Truncate(keep_num);

  CHECK_LE(available_len_, pre_nms_slice.available_len());
}

#define INITIATE_SCORED_BBOX_SLICE(T, type_cpp) template class ScoredBBoxSlice<T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_SCORED_BBOX_SLICE, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
