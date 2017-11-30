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
  explicit Shape(const ShapeProto& shape_proto);
  ~Shape() = default;

  bool operator==(const Shape& rhs) const;
  std::string DebugStr() const;

  void ToProto(ShapeProto*) const;

  template<typename StreamT>
  void SerializeWithTextFormat(StreamT& out_stream) const;

  // Getters and Setters
  const std::vector<int64_t>& dim_vec() const { return dim_vec_; }
  int64_t elem_cnt() const { return elem_cnt_; }
  int64_t At(int64_t index) const { return dim_vec_[index]; }
  void Set(int64_t index, int64_t val);
  int64_t NumAxes() const { return dim_vec_.size(); }
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t Count(int64_t begin_axis) const;

 private:
  void UpdateElemCnt();

  std::vector<int64_t> dim_vec_;
  int64_t elem_cnt_;
};

template<typename StreamT>
void Shape::SerializeWithTextFormat(StreamT& out_stream) const {
  for (int64_t dim : dim_vec_) { out_stream << std::to_string(dim) << ' '; }
}

std::ostream& operator<<(std::ostream& out, const Shape& shape);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHAPE_H_
