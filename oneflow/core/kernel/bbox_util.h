#ifndef ONEFLOW_CORE_KERNEL_BBOX_UTIL_H_
#define ONEFLOW_CORE_KERNEL_BBOX_UTIL_H_

namespace oneflow {

template<typename T, size_t N>
class BBoxElem;

template<template<typename> class Wrapper, typename ElemType, size_t N>
class BBoxElem<Wrapper<ElemType>, N> {
 public:
  using ElemArray = std::array<ElemType, N>;
  using WrapperType = Wrapper<ElemType>;

  OF_DISALLOW_COPY_AND_MOVE(BBoxElem);
  BBoxElem() = delete;
  ~BBoxElem() = delete;

  static const WrapperType* Cast(const ElemType* ptr) {
    return reinterpret_cast<const WrapperType*>(ptr);
  }
  static WrapperType* MutCast(ElemType* ptr) { return reinterpret_cast<WrapperType*>(ptr); }

  const ElemArray& elem() const { return elem_; }
  ElemArray& mut_elem() { return elem_; }

 private:
  ElemArray elem_;
};

template<typename T>
class BBoxDelta;

template<typename T>
class BBox;

template<template<typename> class Impl, typename T>
class BBox<Impl<T>> : public BBoxElem<BBox<T>, 4> {
 public:
  using BBoxArray = typename BBoxElem<BBox<T>, 4>::ElemArray;
  template<typename U>
  using BBoxBase = BBox<Impl<U>>;

  Impl<T>* impl() { return dynamic_cast<Impl<T>>(this); }

  const BBoxArray& bbox() const { return this->elem(); }
  BBoxArray& mut_bbox() { return this->mut_elem(); }

  template<typename U>
  float InterOverUnion(const BBoxBase<U>* other) const {
    const float iw =
        std::min<float>(right(), other->right()) - std::max<float>(left(), other->left()) + 1.f;
    if (iw <= 0) { return 0.f; }
    const float ih =
        std::min<float>(bottom(), other->bottom()) - std::max<float>(top(), other->top()) + 1.f;
    if (ih <= 0) { return 0.f; }
    const float inter = iw * ih;
    return inter / (area() + other->area() - inter);
  }

  template<typename U, typename V>
  void Transform(const BBoxBase<U>* bbox, const BBoxDelta<V>* delta,
                 const BBoxRegressionWeights& bbox_reg_ws) {
    const float dx = delta->dx() / bbox_reg_ws.weight_x();
    const float dy = delta->dy() / bbox_reg_ws.weight_y();
    const float dw = delta->dw() / bbox_reg_ws.weight_w();
    const float dh = delta->dh() / bbox_reg_ws.weight_h();

    const float pred_ctr_x = dx * bbox->weight() + bbox->center_x();
    const float pred_ctr_y = dy * bbox->height() + bbox->center_y();
    const float pred_w = std::exp(dw) * bbox->weight();
    const float pred_h = std::exp(dh) * bbox->height();
    set_center_coord(pred_ctr_x, pred_ctr_y, pred_w, pred_h);
  }

  void Clip(const int64_t height, const int64_t width) {
    const T left = std::max<T>(std::min<T>(left(), width - 1), 0);
    const T top = std::max<T>(std::min<T>(top(), height - 1), 0);
    const T right = std::min<T>(right(), width - 1), 0);
    const T bottom = std::min<T>(bottom(), height - 1), 0);
    set_corner_coord(left, top, right, bottom);
  }

  T left() const { return impl()->left(); }
  T right() const { return impl()->right(); }
  T top() const { return impl()->top(); }
  T bottom() const { return impl()->bottom(); }
  T center_x() const { return impl()->center_x(); }
  T center_y() const { return impl()->center_y(); }
  T width() const { return impl()->width(); }
  T height() const { return impl()->height(); }
  T area() const { return width() * height(); }

  void set_center_coord(T ctr_x, T ctr_y, T w, T h) {
    impl()->set_center_coord(ctr_x, ctr_y, w, h);
  }
  void set_corner_coord(T left, T top, T right, T bottom) {
    impl()->set_corner_coord(left, top, right, bottom);
  }
};

enum class BBoxCoord { kCorner = 0, kCenter };

template<BBoxCoord Coord, typename T>
class BBoxImpl;

template<typename T>
class BBoxImpl<BBoxCoord::kCorner, T> final : public BBox<BBoxImpl<T>> {
 public:
  T left() const { return this->bbox()[0]; }
  T right() const { return this->bbox()[1]; }
  T top() const { return this->bbox()[2]; }
  T bottom() const { return this->bbox()[3]; }
  T center_x() const { return this->left() + 0.5f * this->width(); }
  T center_y() const { return this->top() + 0.5f * this->height(); }
  T width() const { return this->right() - this->left() + OneVal<T>::value; }
  T height() const { return this->bottom() - this->top() + OneVal<T>::value; }
  void set_center_coord(T ctr_x, T ctr_y, T w, T h) {
    this->mut_bbox()[0] = static_cast<T>(ctr_x - 0.5f * w);
    this->mut_bbox()[1] = static_cast<T>(ctr_y - 0.5f * h);
    this->mut_bbox()[2] = static_cast<T>(ctr_x + 0.5f * w - 1.f);
    this->mut_bbox()[3] = static_cast<T>(ctr_y + 0.5f * h - 1.f);
  }
  void set_corner_coord(T left, T top, T right, T bottom) {
    this->mut_bbox()[0] = left;
    this->mut_bbox()[1] = top;
    this->mut_bbox()[2] = right;
    this->mut_bbox()[3] = bottom;
  }
};

template<typename T>
using CornerCoordBBox = BBoxImpl<BBoxCoord::kCorner, T>;

template<typename T>
class BBoxDelta final : public BBoxElem<BBoxDelta<T>, 4> {
 public:
  T dx() const { return this->elem()[0]; }
  T dy() const { return this->elem()[1]; }
  T dw() const { return this->elem()[2]; }
  T dh() const { return this->elem()[3]; }

  void set_dx(T dx) { this->mut_elem()[0] = dx; }
  void set_dy(T dy) { this->mut_elem()[1] = dy; }
  void set_dw(T dw) { this->mut_elem()[2] = dw; }
  void set_dh(T dh) { this->mut_elem()[3] = dh; }

  template<typename U, typename V>
  void TransformInverse(const BBox<U>* bbox, const BBox<V>* target_bbox,
                        const BBoxRegressionWeights& bbox_reg_ws) {
    set_dx(bbox_reg_ws.weight_x() * (target_bbox->center_x() - bbox->center_x()) / bbox->width());
    set_dy(bbox_reg_ws.weight_y() * (target_bbox->center_y() - bbox->center_y()) / bbox->height());
    set_dw(bbox_reg_ws.weight_w() * std::log(target_bbox->width() / bbox->width()));
    set_dh(bbox_reg_ws.weight_h() * std::log(target_bbox->height() / bbox->height()));
  }
};

template<typename T>
class BBoxWeights final : public BBoxElem<BBoxWeights<T>, 4> {
 public:
  inline T weight_x() const { return this->elem()[0]; }
  inline T weight_y() const { return this->elem()[1]; }
  inline T weight_w() const { return this->elem()[2]; }
  inline T weight_h() const { return this->elem()[3]; }

  inline void set_weight_x(T weight_x) { this->mut_elem()[0] = weight_x; }
  inline void set_weight_y(T weight_y) { this->mut_elem()[1] = weight_y; }
  inline void set_weight_w(T weight_w) { this->mut_elem()[2] = weight_w; }
  inline void set_weight_h(T weight_h) { this->mut_elem()[3] = weight_h; }
};

}  // namespace oneflow