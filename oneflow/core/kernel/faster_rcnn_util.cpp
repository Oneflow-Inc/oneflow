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
    FOR_RANGE(int32_t, j, 0, scales_size) {
      const int32_t size = conf.anchor_scales(j) * conf.anchor_scales(j);
      const int32_t w = std::round(std::sqrt(size / conf.aspect_ratios(i)));
      const int32_t h = std::round(w * conf.aspect_ratios(i));
      BBox<T>* cur_anchor_bbox = base_anchor_bbox + i * scales_size + j;
      cur_anchor_bbox->set_x1(std::round(base_ctr - 0.5 * (w - 1)));
      cur_anchor_bbox->set_y1(std::round(base_ctr - 0.5 * (h - 1)));
      cur_anchor_bbox->set_x2(std::round(base_ctr + 0.5 * (w - 1)));
      cur_anchor_bbox->set_y2(std::round(base_ctr + 0.5 * (h - 1)));
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
    converted_gt_bbox[i].set_x2(gt_bbox[i].x2() * image_width);
    converted_gt_bbox[i].set_y2(gt_bbox[i].y2() * image_height);
  }
  return boxes_num;
}

#define INITIATE_FASTER_RCNN_UTIL(T, type_cpp) template struct FasterRcnnUtil<T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_FASTER_RCNN_UTIL, FLOATING_DATA_TYPE_SEQ);

/* BBoxSlice */
template<typename T>
BBoxSlice<T>::BBoxSlice(size_t capacity, const T* boxes_ptr, int32_t* index_ptr,
                        bool init_index = true)
    : bbox_ptr_(bbox_ptr), index_ptr_(index_ptr), capacity_(capacity), size_(0) {
  if (init_index) {
    size_ = capacity;
    std::iota(index_ptr_, index_ptr_ + size_, 0);
  }
}

template<typename T>
void BBoxSlice<T>::Truncate(size_t size) {
  CHECK_GE(size, 0);
  if (size < capacity_) { size_ = size; }
}

template<typename T>
void BBoxSlice<T>::Sort(const std::function<bool(size_t, size_t)>& Compare) {
  std::sort(index_ptr_, index_ptr_ + size_,
            [&](int32_t index_lhs, int32_t index_rhs) { return Compare(index_lhs, index_rhs); });
}

template<typename T>
void BBoxSlice<T>::Sort(const std::function<bool(const BBox<T>&, const BBox<T>&)>& Compare) {
  std::sort(index_ptr_, index_ptr_ + size_, [&](int32_t index_lhs, int32_t index_rhs) {
    const BBox<T>* bbox = BBox<T>::Cast(bbox_ptr_);
    return Compare(bbox[index_lhs], bbox[index_rhs]);
  });
}

template<typename T>
void BBoxSlice<T>::Filter(const std::function<bool(const BBox<T>*)>& FilterMethod) {
  size_t keep_num = 0;
  FOR_RANGE(size_t, i, 0, size_) {
    if (!FilterMethod(GetBBox(i))) {
      // keep_num <= i so index_ptr_ never be written before read
      index_ptr_[keep_num++] = index_ptr_[i];
    }
  }
  size_ = keep_num;
}

template<typename T>
void BBoxSlice<T>::Shuffle(size_t begin, size_t end) {
  CHECK_LE(end, size_);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(index_ptr_ + begin, index_ptr_ + end, gen);
}

#define INITIATE_BBOX_SLICE(T, type_cpp) template class BBoxSlice<T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_BBOX_SLICE, FLOATING_DATA_TYPE_SEQ);

template<typename T, int32_t N>
LabeledBBoxSlice<T, N>::LabeledBBoxSlice(size_t capacity, const T* boxes_ptr, int32_t* label_ptr,
                                         int32_t* index_ptr, bool init_index)
    : capacity_(capacity),
      bbox_ptr_(bbox_ptr),
      label_ptr_(label_ptr),
      index_ptr_(index_ptr),
      size_(0) {
  if (init_index) {
    size_ = capacity;
    std::iota(index_ptr_, index_ptr_ + size_, 0);
  }
  std::fill(group_labels_.begin(), group_labels_.end(), {0, 0, 0});
}

template<typename T, int32_t N>
void LabeledBBoxSlice<T, N>::GroupByLabel() {
  std::sort(index_ptr_, index_ptr_ + size_, [&](int32_t index + lhs, int32_t index_rhs) {
    return label_ptr_[index_lhs] > label_ptr_[index_rhs];
  });
  int32_t last_label =
      label_ptr_[index_ptr_[0]] + 1;  // init last_label to one more than biggest label
  int32_t group_index = -1;
  FOR_RANGE(int32_t, i, 0, size_) {
    int32_t cur_label = label_ptr_[index_ptr_[i]];
    if (cur_label != last_label) {
      const GroupLabel& cur_group_label = group_labels_[++group_index];
      cur_group_label.label = cur_label;
      cur_group_label.begin = i;
      cur_group_label.size = 1;
      last_label = cur_label;
    } else {
      group_labels_[group_index].size += 1;
    }
  }
}

template<typename T, int32_t N>
size_t LabeledBBoxSlice<T, N>::Subsample(int32_t label, size_t sample_num) {
  auto group_label_it =
      std::find_if(group_labels_.begin(), group_labels_.end(),
                   [label](const GroupLabel& group_label) { return group_label.label == label; });
  size_t begin = group_label_it->begin;
  size_t size = group_label_it->size;
  if (size < sample_num) { return size; }
  Shuffle(begin, begin + size);
  FOR_RANGE(size_t, i, begin + sample_num, begin + size) { label_ptr_[index_ptr_[i]] = -1; }
  return sample_num;
}

template<typename T, int32_t N>
int32_t LabeledBBoxSlice<T, N>::GetLabelCount(int32_t label) {
  for (auto it : group_labels_) {
    if (it->label == label) { return it->size; }
  }
  LOG(FATAL) << "The label type is not found";
  return -1;
}

// template<typename T, int32_t N>
// int32_t LabeledBBoxSlice::get_label_start_index(int32_t label) {
//   int32_t start_index = 0;
//   FOR_RANGE(int32_t, 0, i, N) {
//     if(label == label_type_[i]) {
//       return start_index;
//     } else {
//       start_index += label_cnt_[i];
//     }
//   }
//   LOG(FATAL) << "The label type is not found";
//   return -1;
// }

// TODO: fix this macro
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
