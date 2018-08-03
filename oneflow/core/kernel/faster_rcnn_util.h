#ifndef ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_
#define ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_

#include "oneflow/core/common/util.h"

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

  inline T x1() const { return bbox_[0]; }
  inline T y1() const { return bbox_[1]; }
  inline T x2() const { return bbox_[2]; }
  inline T y2() const { return bbox_[3]; }

  inline T& operator[](int32_t i) { return bbox_[i]; }
  inline const T& operator[](int32_t i) const { return bbox_[i]; }

  inline void set_x1(T x1) { bbox_[0] = x1; }
  inline void set_y1(T y1) { bbox_[1] = y1; }
  inline void set_x2(T x2) { bbox_[2] = x2; }
  inline void set_y2(T y2) { bbox_[3] = y2; }

  inline int32_t Area() const { return (x2() - x1() + 1) * (y2() - y1() + 1); }

  inline float InterOverUnion(const BBox<T>* other) const {
    const int32_t iw = std::min(x2(), other->x2()) - std::max(x1(), other->x1()) + 1;
    if (iw <= 0) { return 0; }
    const int32_t ih = std::min(y2(), other->y2()) - std::max(y1(), other->y1()) + 1;
    if (ih <= 0) { return 0; }
    const float inter = iw * ih;
    return inter / (Area() + other->Area() - inter);
  }

  void Transform(const BBox<T>* bbox, const BBoxDelta<T>* delta) {
    const float w = bbox->x2() - bbox->x1() + 1.0f;
    const float h = bbox->y2() - bbox->y1() + 1.0f;
    const float ctr_x = bbox->x1() + 0.5f * w;
    const float ctr_y = bbox->y1() + 0.5f * h;

    const float pred_ctr_x = delta->dx() * w + ctr_x;
    const float pred_ctr_y = delta->dy() * h + ctr_y;
    const float pred_w = std::exp(delta->dw()) * w;
    const float pred_h = std::exp(delta->dh()) * h;

    set_x1(pred_ctr_x - 0.5f * pred_w);
    set_x2(pred_ctr_y - 0.5f * pred_h);
    set_y1(pred_ctr_x + 0.5f * pred_w - 1.f);
    set_y2(pred_ctr_y + 0.5f * pred_h - 1.f);
  }

  void Clip(const int64_t height, const int64_t width) {
    set_x1(std::max<T>(std::min<T>(x1(), width), 0));
    set_x2(std::max<T>(std::min<T>(x2(), height), 0));
    set_y1(std::max<T>(std::min<T>(y1(), width), 0));
    set_y2(std::max<T>(std::min<T>(y2(), height), 0));
  }

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

  void TransformInverse(const BBox<T>* bbox, const BBox<T>* target_bbox) {
    float w = bbox->x2() - bbox->x1() + 1.0f;
    float h = bbox->y2() - bbox->y1() + 1.0f;
    float ctr_x = bbox->x1() + 0.5f * w;
    float ctr_y = bbox->y1() + 0.5f * h;

    float t_w = target_bbox->x2() - target_bbox->x1() + 1.0f;
    float t_h = target_bbox->y2() - target_bbox->y1() + 1.0f;
    float t_ctr_x = target_bbox->x1() + 0.5f * t_w;
    float t_ctr_y = target_bbox->y1() + 0.5f * t_h;

    set_dx((t_ctr_x - ctr_x) / w);
    set_dy((t_ctr_y - ctr_y) / h);
    set_dw(std::log(t_w / w));
    set_dh(std::log(t_h / h));
  }

 private:
  std::array<T, 4> delta_;
};

template<typename T>
struct FasterRcnnUtil final {
  static void BboxTransform(int64_t boxes_num, const T* bboxes, const T* deltas, T* pred_bboxes);
  static void BboxTransformInv(int64_t boxes_num, const T* bboxes, const T* target_bboxes,
                               T* deltas);
  static void ClipBoxes(int64_t boxes_num, const int64_t image_height, const int64_t image_width,
                        T* bboxes);
  static int32_t Nms(const T* img_proposal_ptr, const int32_t* sorted_score_slice_ptr,
                     const int32_t pre_nms_top_n, const int32_t post_nms_top_n,
                     const float nms_threshold, int32_t* area_ptr, int32_t* post_nms_slice_ptr);
  static float InterOverUnion(const BBox<T>& box1, const int32_t area1, const BBox<T>& box2,
                              const int32_t area2);
  static void SortByScore(const int64_t num, const T* score_ptr, int32_t* sorted_score_slice_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_
