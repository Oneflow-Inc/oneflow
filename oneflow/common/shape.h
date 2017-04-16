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
  ~Shape() = default;

  Shape(const std::vector<int64_t>& shape_vec) : shape_vec_(shape_vec) {
    UpdateElemCnt();
  }
  
  bool operator == (const Shape& rhs) const {
    return shape_vec_ == rhs.shape_vec_;
  }

  const std::vector<int64_t>& shape_vec() const { return shape_vec_; }
  int64_t elem_cnt() const { return elem_cnt_; }
  
  std::string ToString() const;

  int64_t shape(int32_t index) const {
    return shape_vec_[CanonicalAxisIndex(index)];
  }
  void set_shape(int32_t index, int64_t val) {
    shape_vec_[CanonicalAxisIndex(index)] = val;
    UpdateElemCnt();
  }


  int32_t NumAxes() const { return shape_vec_.size(); }

  int64_t Count(int32_t start_axis, int32_t end_axis) const;
  int64_t Count(int32_t start_axis) const;
  
  int64_t Num() const;
  int64_t Dim() const;
  int64_t Channel() const;
  int64_t Height() const;
  int64_t Width() const;
  int64_t Offset(const int64_t n,
                 const int64_t c = 0,
                 const int64_t h = 0,
                 const int64_t w = 0) const;

 private:
  void UpdateElemCnt();
  int64_t CanonicalAxisIndex(int32_t axis_index) const;

  std::vector<int64_t> shape_vec_;
  int64_t elem_cnt_;

};

} // namespace oneflow

#endif // ONEFLOW_COMMON_SHAPE_VEC_H_
