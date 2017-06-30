#ifndef ONEFLOW_CORE_COMMON_SHAPE_H_
#define ONEFLOW_CORE_COMMON_SHAPE_H_

#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class Shape final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Shape);
  Shape() : elem_cnt_(0) {}
  explicit Shape(const std::vector<int64_t>& dim_vec);
  Shape(const ShapeProto& shape_proto);
  ~Shape() = default;

  bool operator==(const Shape& rhs) const;
  std::string DebugStr() const;

  void ToProto(ShapeProto*) const;

  // Getters and Setters
  const std::vector<int64_t>& dim_vec() const { return dim_vec_; }
  int64_t elem_cnt() const { return elem_cnt_; }
  int64_t At(int64_t index) const;
  void Set(int64_t index, int64_t val);
  int64_t NumAxes() const { return dim_vec_.size(); }
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t Count(int64_t begin_axis) const;

 private:
  void UpdateElemCnt();
  int64_t CanonicalAxisIndex(int64_t axis_index) const;

  std::vector<int64_t> dim_vec_;
  int64_t elem_cnt_;
};

std::ostream& operator<<(std::ostream& out, const Shape& shape);

inline Shape::Shape(const std::vector<int64_t>& dim_vec) : dim_vec_(dim_vec) {
  UpdateElemCnt();
}

inline bool Shape::operator==(const Shape& rhs) const {
  return dim_vec_ == rhs.dim_vec_;
}

inline int64_t Shape::At(int64_t index) const {
  return dim_vec_[CanonicalAxisIndex(index)];
}

inline void Shape::Set(int64_t index, int64_t val) {
  dim_vec_[CanonicalAxisIndex(index)] = val;
  UpdateElemCnt();
}

inline int64_t Shape::Count(int64_t begin_axis) const {
  return Count(begin_axis, NumAxes());
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHAPE_H_
