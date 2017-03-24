#include "common/shape.h"
namespace oneflow {
Shape::Shape(const BlobShape& proto_shape) {
  CHECK_EQ(proto_shape.dim_size(), 4)
    << "any shape must be in 4 dimensions";
  count_ = 1;
  for (int32_t d = 0; d < proto_shape.dim_size(); ++d) {
    int64_t val = proto_shape.dim(d);
    shape_.push_back(val);
    count_ *= val;
  }
}

Shape::Shape(int64_t num,
  int64_t channels,
  int64_t height,
  int64_t width) {
  shape_.push_back(num);
  shape_.push_back(channels);
  shape_.push_back(height);
  shape_.push_back(width);
  count_ = num * channels * height * width;
}

Shape::Shape(int64_t num,
  int64_t dim) {
  shape_.push_back(num);
  shape_.push_back(dim);
  count_ = num * dim;
}

}  // namespace oneflow
