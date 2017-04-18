#ifndef ONEFLOW_COMMON_SHAPE_VEC_H_
#define ONEFLOW_COMMON_SHAPE_VEC_H_

#include <vector>
#include <string>
#include "common/util.h"

namespace oneflow {

class Shape final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Shape);
  Shape() : elem_cnt_(0) {}
  Shape(const std::vector<int64_t>& shape_vec);
  ~Shape() = default;
  
  bool operator == (const Shape& rhs) const;
  std::string ToString() const;

  // Getters and Setters
  const std::vector<int64_t>& shape_vec() const { return shape_vec_; }
  int64_t elem_cnt() const { return elem_cnt_; }
  int64_t At(int32_t index) const;
  void Set(int32_t index, int64_t val);
  int32_t NumAxes() const { return shape_vec_.size(); }
  int64_t Count(int32_t start_axis, int32_t end_axis) const;
  int64_t Count(int32_t start_axis) const;

 private:
  void UpdateElemCnt();
  int64_t CanonicalAxisIndex(int32_t axis_index) const;

  std::vector<int64_t> shape_vec_;
  int64_t elem_cnt_;

};

inline Shape::Shape(const std::vector<int64_t>& shape_vec) :
    shape_vec_(shape_vec) {
  UpdateElemCnt();
}

inline bool Shape::operator == (const Shape& rhs) const {
  return shape_vec_ == rhs.shape_vec_;
}

inline int64_t Shape::At(int32_t index) const {
  return shape_vec_[CanonicalAxisIndex(index)];
}

inline void Shape::Set(int32_t index, int64_t val) {
  shape_vec_[CanonicalAxisIndex(index)] = val;
  UpdateElemCnt();
}

inline int64_t Shape::Count(int32_t start_axis) const {
  return Count(start_axis, NumAxes());
}

} // namespace oneflow

#endif // ONEFLOW_COMMON_SHAPE_VEC_H_
