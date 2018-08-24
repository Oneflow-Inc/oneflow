#ifndef ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_
#define ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

template<typename T>
class BBoxDelta;

template<typename T>
class BBox final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BBox);
  BBox() = delete;
  ~BBox() = delete;

  static const BBox* Cast(const T* ptr) { return reinterpret_cast<const BBox*>(ptr); }
  static BBox* MutCast(T* ptr) { return reinterpret_cast<BBox*>(ptr); }

  const std::array<T, 4>& bbox() const { return bbox_; }
  std::array<T, 4>& mut_bbox() { return bbox_; }

  inline T x1() const { return bbox_[0]; }
  inline T y1() const { return bbox_[1]; }
  inline T x2() const { return bbox_[2]; }
  inline T y2() const { return bbox_[3]; }
  inline void set_x1(T x1) { bbox_[0] = x1; }
  inline void set_y1(T y1) { bbox_[1] = y1; }
  inline void set_x2(T x2) { bbox_[2] = x2; }
  inline void set_y2(T y2) { bbox_[3] = y2; }

  inline T width() const { return x2() - x1() + static_cast<T>(1); }
  inline T height() const { return y2() - y1() + static_cast<T>(1); }
  inline T Area() const { return width() * height(); }
  inline float InterOverUnion(const BBox* other) const {
    const float iw = std::min<float>(x2(), other->x2()) - std::max<float>(x1(), other->x1()) + 1.f;
    if (iw <= 0) { return 0; }
    const float ih = std::min<float>(y2(), other->y2()) - std::max<float>(y1(), other->y1()) + 1.f;
    if (ih <= 0) { return 0; }
    const float inter = iw * ih;
    return inter / (Area() + other->Area() - inter);
  }

  void Transform(const BBox<T>* bbox, const BBoxDelta<T>* delta,
                 const BBoxRegressionWeights& bbox_reg_ws) {
    const float w = bbox->x2() - bbox->x1() + 1.0f;
    const float h = bbox->y2() - bbox->y1() + 1.0f;
    const float ctr_x = bbox->x1() + 0.5f * w;
    const float ctr_y = bbox->y1() + 0.5f * h;

    const float dx = delta->dx() / bbox_reg_ws.weight_x();
    const float dy = delta->dy() / bbox_reg_ws.weight_y();
    const float dw = delta->dw() / bbox_reg_ws.weight_w();
    const float dh = delta->dh() / bbox_reg_ws.weight_h();

    const float pred_ctr_x = dx * w + ctr_x;
    const float pred_ctr_y = dy * h + ctr_y;
    const float pred_w = std::exp(dw) * w;
    const float pred_h = std::exp(dh) * h;

    set_x1(pred_ctr_x - 0.5f * pred_w);
    set_y1(pred_ctr_y - 0.5f * pred_h);
    set_x2(pred_ctr_x + 0.5f * pred_w - 1.f);
    set_y2(pred_ctr_y + 0.5f * pred_h - 1.f);
  }

  void Clip(const int64_t height, const int64_t width) {
    set_x1(std::max<T>(std::min<T>(x1(), width - 1), 0));
    set_y1(std::max<T>(std::min<T>(y1(), height - 1), 0));
    set_x2(std::max<T>(std::min<T>(x2(), width - 1), 0));
    set_y2(std::max<T>(std::min<T>(y2(), height - 1), 0));
  }

 private:
  std::array<T, 4> bbox_;
};

template<typename T>
class BBoxWeights final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BBoxWeights);
  BBoxWeights() = delete;
  ~BBoxWeights() = delete;

  static const BBoxWeights* Cast(const T* ptr) { return reinterpret_cast<const BBoxWeights*>(ptr); }
  static BBoxWeights* MutCast(T* ptr) { return reinterpret_cast<BBoxWeights*>(ptr); }

  inline T weight_x() const { return weights_[0]; }
  inline T weight_y() const { return weights_[1]; }
  inline T weight_w() const { return weights_[2]; }
  inline T weight_h() const { return weights_[3]; }

  inline void set_weight_x(T weight_x) { weights_[0] = weight_x; }
  inline void set_weight_y(T weight_y) { weights_[1] = weight_y; }
  inline void set_weight_w(T weight_w) { weights_[2] = weight_w; }
  inline void set_weight_h(T weight_h) { weights_[3] = weight_h; }

 private:
  std::array<T, 4> weights_;
};

template<typename T>
class BBoxDelta final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BBoxDelta);
  BBoxDelta() = delete;
  ~BBoxDelta() = delete;

  static const BBoxDelta* Cast(const T* ptr) { return reinterpret_cast<const BBoxDelta*>(ptr); }
  static BBoxDelta* MutCast(T* ptr) { return reinterpret_cast<BBoxDelta*>(ptr); }

  inline T dx() const { return delta_[0]; }
  inline T dy() const { return delta_[1]; }
  inline T dw() const { return delta_[2]; }
  inline T dh() const { return delta_[3]; }

  inline void set_dx(T dx) { delta_[0] = dx; }
  inline void set_dy(T dy) { delta_[1] = dy; }
  inline void set_dw(T dw) { delta_[2] = dw; }
  inline void set_dh(T dh) { delta_[3] = dh; }

  void TransformInverse(const BBox<T>* bbox, const BBox<T>* target_bbox,
                        const BBoxRegressionWeights& bbox_reg_ws) {
    float w = bbox->x2() - bbox->x1() + 1.0f;
    float h = bbox->y2() - bbox->y1() + 1.0f;
    float ctr_x = bbox->x1() + 0.5f * w;
    float ctr_y = bbox->y1() + 0.5f * h;

    float t_w = target_bbox->x2() - target_bbox->x1() + 1.0f;
    float t_h = target_bbox->y2() - target_bbox->y1() + 1.0f;
    float t_ctr_x = target_bbox->x1() + 0.5f * t_w;
    float t_ctr_y = target_bbox->y1() + 0.5f * t_h;

    set_dx(bbox_reg_ws.weight_x() * (t_ctr_x - ctr_x) / w);
    set_dy(bbox_reg_ws.weight_y() * (t_ctr_y - ctr_y) / h);
    set_dw(bbox_reg_ws.weight_w() * std::log(t_w / w));
    set_dh(bbox_reg_ws.weight_h() * std::log(t_h / h));
  }

 private:
  std::array<T, 4> delta_;
};

class Slice {
 public:
  Slice(size_t capacity, int32_t* index_ptr, bool init_index = true)
      : capacity_(capacity), size_(capacity), index_ptr_(index_ptr) {
    if (init_index) { std::iota(index_ptr_, index_ptr_ + size_, 0); }
  }

  void Truncate(size_t size) {
    CHECK_GE(size, 0);
    if (size < capacity_) { size_ = size; }
  }

  void Sort(const std::function<bool(int32_t, int32_t)>& Compare) {
    std::sort(index_ptr_, index_ptr_ + size_,
              [&](int32_t lhs_index, int32_t rhs_index) { return Compare(lhs_index, rhs_index); });
  }

  void Shuffle(size_t begin, size_t end) {
    CHECK_LE(end, size_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(index_ptr_ + begin, index_ptr_ + end, gen);
  }

  void Shuffle() { Shuffle(0, size_); }

  int32_t GetIndex(size_t n) const {
    CHECK_LT(n, size_);
    return index_ptr_[n];
  }

  size_t capacity() const { return capacity_; }
  size_t size() const { return size_; };
  const int32_t* index_ptr() const { return index_ptr_; }
  int32_t* mut_index_ptr() { return index_ptr_; }

 private:
  const size_t capacity_;
  size_t size_;
  int32_t* index_ptr_;
};

template<typename T>
class BoxesSlice : public Slice {
 public:
  BoxesSlice(size_t capacity, int32_t* index_ptr, const T* boxes_ptr, bool init_index = true)
      : Slice(capacity, index_ptr, init_index), boxes_ptr_(boxes_ptr) {}
  BoxesSlice(const Slice& slice, const T* boxes_ptr) : Slice(slice), boxes_ptr_(boxes_ptr) {}

  void FilterByBox(const std::function<bool(const BBox<T>*)>& FilterBox) {
    size_t keep_num = 0;
    FOR_RANGE(size_t, i, 0, this->size()) {
      if (!FilterBox(GetBBox(i))) {
        // keep_num <= i so index_ptr_ never be written before read
        this->mut_index_ptr()[keep_num++] = this->index_ptr()[i];
      }
    }
    this->Truncate(keep_num);
  }

  void SortByBox(const std::function<bool(const BBox<T>&, const BBox<T>&)>& Compare) {
    std::sort(this->mut_index_ptr(), this->mut_index_ptr() + this->size(),
              [&](int32_t lhs_index, int32_t rhs_index) {
                const BBox<T>* bbox = BBox<T>::Cast(boxes_ptr_);
                return Compare(bbox[lhs_index], bbox[rhs_index]);
              });
  }

  inline const BBox<T>* GetBBox(size_t n) const {
    CHECK_LT(n, this->size());
    return BBox<T>::Cast(boxes_ptr_) + this->GetIndex(n);
  }

  const T* boxes_ptr() const { return boxes_ptr_; }
  const BBox<T>* bbox(int32_t box_index) const { return BBox<T>::Cast(boxes_ptr_) + box_index; }

 private:
  const T* boxes_ptr_;
};

struct GroupLabel {
  int32_t label;
  size_t begin;
  size_t size;
};

template<typename SliceType, size_t N>
class LabeledBoxesSlice : public SliceType {
 public:
  LabeledBoxesSlice(const SliceType& slice, int32_t* label_ptr, bool init_label = true)
      : SliceType(slice), label_ptr_(label_ptr) {
    GroupLabel group_label{0, 0, 0};
    std::fill(group_labels_.begin(), group_labels_.end(), group_label);
    if (init_label) { std::fill(label_ptr_, label_ptr_ + this->capacity(), -1); }
  }

  void GroupByLabel() {
    int32_t* index_ptr = this->mut_index_ptr();
    size_t size = this->size();
    std::sort(index_ptr, index_ptr + size, [&](int32_t index_lhs, int32_t index_rhs) {
      return label_ptr_[index_lhs] > label_ptr_[index_rhs];
    });
    // init last_label to one more than biggest label
    int32_t last_label = label_ptr_[index_ptr[0]] + 1;
    int32_t group_index = -1;
    FOR_RANGE(int32_t, i, 0, size) {
      int32_t cur_label = label_ptr_[index_ptr[i]];
      if (cur_label != last_label) {
        GroupLabel& cur_group_label = group_labels_[++group_index];
        cur_group_label.label = cur_label;
        cur_group_label.begin = i;
        cur_group_label.size = 1;
        last_label = cur_label;
      } else {
        group_labels_[group_index].size += 1;
      }
    }
  }

  size_t Subsample(int32_t label, size_t sample_num,
                   const std::function<void(int32_t)>& EnableHandler,
                   const std::function<void(int32_t)>& DisableHandler) {
    auto group_label_it =
        std::find_if(group_labels_.begin(), group_labels_.end(),
                     [label](const GroupLabel& group_label) { return group_label.label == label; });
    size_t begin = group_label_it->begin;
    size_t size = group_label_it->size;
    if (size > sample_num) {
      this->Shuffle(begin, begin + size);
    } else {
      sample_num = size;
    }
    FOR_RANGE(size_t, i, begin, begin + size) {
      int32_t cur_box_index = this->GetIndex(i);
      if (i < sample_num) {
        EnableHandler(cur_box_index);
      } else {
        DisableHandler(cur_box_index);
      }
    }
    return sample_num;
  }

  const int32_t* label_ptr() const { return label_ptr_; }
  int32_t* mut_label_ptr() { return label_ptr_; }
  int32_t label(int32_t box_index) const { return label_ptr_[box_index]; }
  void set_label(int32_t box_index, int32_t label) { label_ptr_[box_index] = label; }

 private:
  int32_t* label_ptr_;
  std::array<GroupLabel, N> group_labels_;
};

template<size_t N, typename SliceType>
LabeledBoxesSlice<SliceType, N> GenLabeledBoxesSlice(const SliceType& slice, int32_t* label_ptr) {
  return LabeledBoxesSlice<SliceType, N>(slice, label_ptr);
}

template<typename SliceType>
class BoxesToNearestGtBoxesSlice : public SliceType {
 public:
  BoxesToNearestGtBoxesSlice(const SliceType& slice, float* max_overlap_ptr,
                             int32_t* gt_box_index_ptr)
      : SliceType(slice),
        max_overlap_ptr_(max_overlap_ptr),
        max_overlap_gt_box_index_ptr_(gt_box_index_ptr) {}

  float GetMaxOverlap(size_t n) { return max_overlap_ptr_[this->GetIndex(n)]; }

  float GetNearestGtBoxIndex(size_t n) { return max_overlap_gt_box_index_ptr_[this->GetIndex(n)]; }

  void UpdateMaxOverlapGtBox(int32_t box_index, int32_t gt_box_index, float overlap,
                             const std::function<void()>& DoUpdateHandle) {
    if (overlap >= max_overlap_ptr_[box_index]) {
      max_overlap_ptr_[box_index] = overlap;
      max_overlap_gt_box_index_ptr_[box_index] = gt_box_index;
      DoUpdateHandle();
    }
  }

  void SortByOverlap(const std::function<bool(float, float)>& Compare) {
    std::sort(this->mut_index_ptr(), this->mut_index_ptr() + this->size(),
              [&](int32_t lhs_index, int32_t rhs_index) {
                return Compare(max_overlap_ptr_[lhs_index], max_overlap_ptr_[rhs_index]);
              });
  }

  int32_t FindByOverlap(const std::function<bool(float)>& Condition) {
    FOR_RANGE(int32_t, i, 0, this->size()) {
      if (Condition(max_overlap_ptr_[this->GetIndex(i)])) { return i; }
    }
    return -1;
  }

  int32_t max_overlap_gt_box_index(int32_t box_index) const {
    return max_overlap_gt_box_index_ptr_[box_index];
  }

 private:
  float* max_overlap_ptr_;
  int32_t* max_overlap_gt_box_index_ptr_;
};

template<typename SliceType>
BoxesToNearestGtBoxesSlice<SliceType> GenBoxesToNearestGtBoxesSlice(const SliceType& slice,
                                                                    float* max_overlap_ptr,
                                                                    int32_t* gt_box_index_ptr) {
  return BoxesToNearestGtBoxesSlice<SliceType>(slice, max_overlap_ptr, gt_box_index_ptr);
}

template<typename SliceType>
class GtBoxesToNearestBoxesSlice : public SliceType {
 public:
  GtBoxesToNearestBoxesSlice(const SliceType& slice, float* gt_max_overlaps_ptr,
                             int32_t* nearest_boxes_index_ptr)
      : SliceType(slice),
        gt_max_overlaps_ptr_(gt_max_overlaps_ptr),
        nearest_boxes_index_ptr_(nearest_boxes_index_ptr),
        last_gt_box_index_(-1),
        last_gt_box_nearest_boxes_index_end_(0),
        nearest_boxes_count_(0) {}

  void UpdateNearestBox(int32_t gt_box_index, int32_t box_index, float overlap) {
    if (gt_box_index != last_gt_box_index_) {
      last_gt_box_nearest_boxes_index_end_ = nearest_boxes_count_;
      last_gt_box_index_ = gt_box_index;
    }
    if (overlap >= gt_max_overlaps_ptr_[gt_box_index]) {
      if (overlap > gt_max_overlaps_ptr_[gt_box_index]) {
        nearest_boxes_count_ = last_gt_box_nearest_boxes_index_end_;
      }
      nearest_boxes_index_ptr_[nearest_boxes_count_++] = box_index;
      gt_max_overlaps_ptr_[gt_box_index] = overlap;
    }
  }

  void ForEachNearestBox(const std::function<void(int32_t)>& Handler) const {
    FOR_RANGE(size_t, i, 0, nearest_boxes_count_) { Handler(nearest_boxes_index_ptr_[i]); }
  }

 private:
  float* gt_max_overlaps_ptr_;
  int32_t* nearest_boxes_index_ptr_;
  int32_t last_gt_box_index_;
  int32_t last_gt_box_nearest_boxes_index_end_;
  size_t nearest_boxes_count_;
};

template<typename SliceType>
GtBoxesToNearestBoxesSlice<SliceType> GenGtBoxesToNearestBoxesSlice(
    const SliceType& slice, float* gt_max_overlaps_ptr, int32_t* nearest_boxes_index_ptr) {
  return GtBoxesToNearestBoxesSlice<SliceType>(slice, gt_max_overlaps_ptr, nearest_boxes_index_ptr);
}

template<typename T>
class ScoredBBoxSlice final {
 public:
  ScoredBBoxSlice(int32_t len, const T* bbox_ptr, const T* score_ptr, int32_t* index_slice)
      : len_(len),
        bbox_ptr_(bbox_ptr),
        score_ptr_(score_ptr),
        index_slice_(index_slice),
        available_len_(len) {}

  void Sort(const std::function<bool(const T, const T, const BBox<T>&, const BBox<T>&)>& Compare);
  void DescSortByScore(bool init_index);
  void DescSortByScore() { DescSortByScore(true); }
  void NmsFrom(float nms_threshold, const ScoredBBoxSlice<T>& pre_nms_slice);

  void Truncate(int32_t len);
  void TruncateByThreshold(float thresh);
  int32_t FindByThreshold(const float thresh);
  void Concat(const ScoredBBoxSlice& other);
  void Filter(const std::function<bool(const T, const BBox<T>*)>& IsFiltered);
  ScoredBBoxSlice<T> Slice(const int32_t begin, const int32_t end);
  void Shuffle();

  inline int32_t GetSlice(int32_t i) const {
    CHECK_LT(i, available_len_);
    return index_slice_[i];
  }
  inline const BBox<T>* GetBBox(int32_t i) const {
    CHECK_LT(i, available_len_);
    return BBox<T>::Cast(bbox_ptr_) + index_slice_[i];
  }
  inline T GetScore(int32_t i) const {
    CHECK_LT(i, available_len_);
    return score_ptr_[index_slice_[i]];
  }

  // Getters
  int32_t len() const { return len_; }
  const T* bbox_ptr() const { return bbox_ptr_; }
  const T* score_ptr() const { return score_ptr_; }
  const int32_t* index_slice() const { return index_slice_; }
  int32_t available_len() const { return available_len_; }
  // Setters
  int32_t* mut_index_slice() { return index_slice_; }

 private:
  const int32_t len_;
  const T* bbox_ptr_;
  const T* score_ptr_;
  int32_t* index_slice_;
  int32_t available_len_;
};

template<typename T>
struct FasterRcnnUtil final {
  static void GenerateAnchors(const AnchorGeneratorConf& conf, Blob* anchors_blob);
  static void BboxTransform(int64_t boxes_num, const T* bboxes, const T* deltas,
                            const BBoxRegressionWeights& bbox_reg_ws, T* pred_bboxes);
  static void BboxTransformInv(int64_t boxes_num, const T* bboxes, const T* target_bboxes,
                               const BBoxRegressionWeights& bbox_reg_ws, T* deltas);
  static void ClipBoxes(int64_t boxes_num, const int64_t image_height, const int64_t image_width,
                        T* bboxes);
  static size_t ConvertGtBoxesToAbsoluteCoord(const FloatList16* gt_boxes,
                                              const size_t image_height, const size_t image_width,
                                              T* converted_gt_boxes);
  static void ForEachOverlapBetweenBoxesAndGtBoxes(
      const BoxesSlice<T>& boxes_slice, const BoxesSlice<T>& gt_boxes_slice,
      const std::function<void(int32_t, int32_t, float)>& Handler);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_
