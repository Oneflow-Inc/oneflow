#ifndef ONEFLOW_BLOB_shape_vec_H_
#define ONEFLOW_BLOB_shape_vec_H_

#include <vector>
#include <string>
#include "common/util.h"

namespace oneflow {

class Shape {
 public:
  // DISALLOW_COPY_AND_MOVE(Shape);
  Shape() = default;
  ~Shape() = default;

  void init(const std::vector<int64_t>& shape_vec);
  
  bool operator == (const Shape& rhs) const {
    return shape_vec_ == rhs.shape_vec_;
  }

  std::string ToString() const;

  int64_t shape(int32_t index) const {
    return shape_vec_[CanonicalAxisIndex(index)];
  }
  void set_shape(int32_t index, int64_t val) {
    shape_vec_[CanonicalAxisIndex(index)] = val;
    UpdateElemCnt();
  }
  const std::vector<int64_t>& shape_vec() const { return shape_vec_; }
  int64_t elem_cnt() const { return elem_cnt_; }

  int32_t NumAxes() const { return shape_vec_.size(); }
  int64_t CanonicalAxisIndex(int32_t axis_index) const;

  int64_t count(int32_t start_axis, int32_t end_axis) const;
  int64_t count(int32_t start_axis) const;
  
  int64_t num() const;
  int64_t dim() const;
  int64_t channels() const;
  int64_t height() const;
  int64_t width() const;
  int64_t offset(const int64_t n,
                 const int64_t c = 0,
                 const int64_t h = 0,
                 const int64_t w = 0) const;

 private:
  void UpdateElemCnt();

  std::vector<int64_t> shape_vec_;
  int64_t elem_cnt_;

};

} // namespace oneflow

#endif // ONEFLOW_BLOB_shape_vec_H_
