#ifndef ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_
#define ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T>
class BBox final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BBox);
  BBox() = delete;
  ~BBox() = delete;

  static const BBox* Cast(const T* ptr) { return reinterpret_cast<const BBox*>(ptr); }
  static BBox* MutCast(T* ptr) { return reinterpret_cast<BBox*>(ptr); }

  inline int32_t x1() const { return bbox_[0]; }
  inline int32_t y1() const { return bbox_[1]; }
  inline int32_t x2() const { return bbox_[2]; }
  inline int32_t y2() const { return bbox_[3]; }

  inline T Area() const { return (x2() - x1() + 1) * (y2() - y1() + 1); }

  inline float InterOverUnion(const BBox* other) const {
    const int32_t iw = std::min(x2(), other->x2()) - std::max(x1(), other->x1()) + 1;
    if (iw <= 0) { return 0; }
    const int32_t ih = std::min(y2(), other->y2()) - std::max(y1(), other->y1()) + 1;
    if (ih <= 0) { return 0; }
    const float inter = iw * ih;
    return inter / (Area() + other->Area() - inter);
  }

  inline void set_x1(T x1) { bbox_[0] = x1; }
  inline void set_y1(T y1) { bbox_[1] = y1; }
  inline void set_x2(T x2) { bbox_[2] = x2; }
  inline void set_y2(T y2) { bbox_[3] = y2; }

 private:
  std::array<T, 4> bbox_;
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

 private:
  std::array<T, 4> delta_;
};

template<typename T>
class ScoredBBoxSlice final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScoredBBoxSlice);
  ScoredBBoxSlice(int32_t len, const T* bbox_ptr, const T* score_ptr)
      : len_(len), bbox_ptr_(bbox_ptr), score_ptr_(score_ptr), available_len_(len) {}

  void SortByScore(bool init_index);
  void SortByScore() { SortByScore(true); }
  void Nms(float nms_threshold, ScoredBBoxSlice* post_nms_slice) const;
  void Truncate(int64_t len) {
    CHECK_GE(len, 0);
    if (len < available_len_) { available_len_ = len; }
  }
  inline const BBox<T>* GetBBox(int64_t i) const {
    return BBox<T>::Cast(bbox_ptr_) + index_slice_[i];
  }

  // Getters
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
  static void BboxTransform(const T* bbox, const T* deltas, T* bbox_pred);
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
