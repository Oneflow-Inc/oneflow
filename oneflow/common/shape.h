#ifndef _COMMON_SHAPE_H_
#define _COMMON_SHAPE_H_
#include <cstdint>
#include <vector>
#include <string>
#include <glog/logging.h>
#include "proto/oneflow.pb.h"

// NOTE(jiyuan): basically borrow from oneflow blob.hpp, functions have exactly
// the same meaning.
namespace oneflow {
class Shape {
 public:
  Shape() : count_(0) {}

  Shape(const Shape& other) : count_(other.count_), shape_(other.shape_) {}
  explicit Shape(const BlobShape& proto_shape);
  Shape(int64_t num, int64_t channels, int64_t height, int64_t width);
  Shape(int64_t num, int64_t dim);
  ~Shape() {}
  void Reshape(const std::vector<int64_t>& shape);
  Shape& operator=(const Shape& other) {
    count_ = other.count_;
    shape_ = other.shape_;
    return *this;
  }
  bool operator==(const Shape& other) const {
    return (shape_ == other.shape_);
  }

  std::string shape_string() const;
  int64_t shape(int32_t index) const;
  void set_shape(int32_t index, int64_t val);
  const std::vector<int64_t>& shape() const;
  int32_t num_axes() const;
  int64_t count() const;
  int64_t CanonicalAxisIndex(int32_t axis_index) const;
  int64_t count(int32_t start_axis, int32_t end_axis) const;
  int64_t count(int32_t start_axis) const;
  int64_t num() const;
  int64_t dim() const;
  int64_t channels() const;
  int64_t height() const;
  int64_t width() const;
  int64_t offset(const int64_t n, const int64_t c = 0, const int64_t h = 0,
    const int64_t w = 0) const;
  void Reshape(int64_t num, int64_t channels, int64_t height,
    int64_t width);
  bool ShapeEquals(const Shape& other) {
    LOG(FATAL) << "Not implemented";
  }

 private:
  std::vector<int64_t> shape_;
  int64_t count_;
};

inline std::string Shape::shape_string() const {
  std::ostringstream stream;
  for (int32_t i = 0; i < shape_.size(); ++i) {
    stream << shape_[i] << " ";
  }
  stream << "(" << count_ << ")";
  return stream.str();
}

inline int64_t Shape::shape(int32_t index) const {
  return shape_[CanonicalAxisIndex(index)];
}
inline void Shape::set_shape(int32_t index, int64_t val) {
  shape_[CanonicalAxisIndex(index)] = val;
  count_ = 1;
  for (auto dim : shape_) {
    count_ *= dim;
  }
}
inline const std::vector<int64_t>& Shape::shape() const {
  return shape_;
}
inline int32_t Shape::num_axes() const {
  return shape_.size();
}
inline int64_t Shape::count() const {
  return count_;
}
inline int64_t Shape::CanonicalAxisIndex(int32_t axis_index) const {
  CHECK_GE(axis_index, -num_axes())
    << "axis " << axis_index << " out of range for " << num_axes()
    << " -D Blob with shape " << shape_string();
  CHECK_LT(axis_index, num_axes())
    << "axis " << axis_index << " out of range for " << num_axes()
    << " -D Blob with shape " << shape_string();
  if (axis_index < 0) {
    return axis_index + num_axes();
  }
  return axis_index;
}
inline int64_t Shape::count(int32_t start_axis, int32_t end_axis) const {
  CHECK_LE(start_axis, end_axis);
  CHECK_GE(start_axis, 0);
  CHECK_GE(end_axis, 0);
  CHECK_LE(start_axis, num_axes());
  CHECK_LE(end_axis, num_axes());
  int64_t count = 1;
  for (int32_t i = start_axis; i < end_axis; ++i) {
    count *= shape(i);
  }
  return count;
}
inline int64_t Shape::count(int32_t start_axis) const {
  return count(start_axis, num_axes());
}
inline int64_t Shape::num() const {
  CHECK_GE(shape_.size(), 1);
  return shape_[0];
}
inline int64_t Shape::dim() const{
  CHECK_EQ(shape_.size(), 2);
  return shape_[1];
}
inline int64_t Shape::channels() const {
  CHECK_EQ(4, shape_.size());
  return shape_[1];
}
inline int64_t Shape::height() const {
  CHECK_EQ(4, shape_.size());
  return shape_[2];
}
inline int64_t Shape::width() const {
  CHECK_EQ(4, shape_.size());
  return shape_[3];
}
inline int64_t Shape::offset(const int64_t n,
  const int64_t c,
  const int64_t h,
  const int64_t w ) const {
  CHECK_GE(n, 0);
  CHECK_LT(n, num());
  CHECK_GE(c, 0);
  CHECK_LT(c, channels());
  CHECK_GE(h, 0);
  CHECK_LT(h, height());
  CHECK_GE(w, 0);
  CHECK_LT(w, width());
  return ((n * channels() + c) * height() + h) * width() + w;
}
inline  void Shape::Reshape(int64_t num,
    int64_t channels,
    int64_t height,
    int64_t width) {
    shape_.clear();
    shape_.push_back(num);
    shape_.push_back(channels);
    shape_.push_back(height);
    shape_.push_back(width);
    count_ = num * channels * height * width;
}
inline void Shape::Reshape(const std::vector<int64_t>& shape) {
  shape_.clear();
  shape_ = shape;
  count_ = 1;
  for (auto dim : shape_) {
    count_ *= dim;
  }
}
}  // namespace oneflow
#endif  // _COMMON_SHAPE_H_
