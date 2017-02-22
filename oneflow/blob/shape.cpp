#include "blob/shape.h"
#include "glog/logging.h"

namespace oneflow {

void Shape::init(const std::vector<int64_t>& shape_vec) {
  shape_vec_ = shape_vec;
  UpdateElemCnt();
}

std::string Shape::ToString() const {
  std::ostringstream stream;
  for (int64_t dim : shape_vec_) {
    stream << dim << " ";
  }
  stream << "(" << elem_cnt_ << ")";
  return stream.str();
}

int64_t Shape::CanonicalAxisIndex(int32_t axis_index) const {
  CHECK_GE(axis_index, -NumAxes())
    << "axis " << axis_index << " out of range for " << NumAxes()
    << " -D Blob with shape " << ToString();
  CHECK_LT(axis_index, NumAxes())
    << "axis " << axis_index << " out of range for " << NumAxes()
    << " -D Blob with shape " << ToString();
  return (axis_index + NumAxes()) % NumAxes();
}

int64_t Shape::count(int32_t start_axis, int32_t end_axis) const {
  CHECK_LE(start_axis, end_axis);
  CHECK_GE(start_axis, 0);
  CHECK_GE(end_axis, 0);
  CHECK_LE(start_axis, NumAxes());
  CHECK_LE(end_axis, NumAxes());
  int64_t count = 1;
  for (int32_t i = start_axis; i < end_axis; ++i) {
    count *= shape(i);
  }
  return count;
}

int64_t Shape::count(int32_t start_axis) const {
  return count(start_axis, NumAxes());
}

int64_t Shape::num() const {
  CHECK_GE(shape_vec_.size(), 1);
  return shape_vec_[0];
}

int64_t Shape::dim() const{
  CHECK_EQ(shape_vec_.size(), 2);
  return shape_vec_[1];
}

int64_t Shape::channels() const {
  CHECK_EQ(4, shape_vec_.size());
  return shape_vec_[1];
}

int64_t Shape::height() const {
  CHECK_EQ(4, shape_vec_.size());
  return shape_vec_[2];
}

int64_t Shape::width() const {
  CHECK_EQ(4, shape_vec_.size());
  return shape_vec_[3];
}

int64_t Shape::offset(const int64_t n,
                      const int64_t c,
                      const int64_t h,
                      const int64_t w) const {
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

void Shape::UpdateElemCnt() {
  elem_cnt_ = 1;
  for (auto dim : shape_vec_) {
    elem_cnt_ *= dim;
  }
}

} // namespace oneflow
