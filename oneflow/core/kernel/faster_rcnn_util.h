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

  inline int32_t width() const { return static_cast<int32_t>(x2() - x1() + 1); }
  inline int32_t height() const { return static_cast<int32_t>(y2() - y1() + 1); }
  inline int32_t Area() const { return width() * height(); }
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
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_FASTER_RCNN_UTIL_H_
