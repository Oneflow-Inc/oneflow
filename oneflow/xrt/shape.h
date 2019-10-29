#ifndef ONEFLOW_XRT_XRT_SHAPE_H_
#define ONEFLOW_XRT_XRT_SHAPE_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace xrt {

class XrtShape {
 public:
  XrtShape() = default;

  explicit XrtShape(const Shape &shape) : shape_(shape) {}

  explicit XrtShape(const Shape &shape, const DataType &dtype)
      : shape_(shape), data_type_(dtype) {}

  const Shape &shape() const { return shape_; }

  const DataType &data_type() const { return data_type_; }

 private:
  Shape shape_;
  DataType data_type_;
};

inline bool operator==(const XrtShape &lhs, const XrtShape &rhs) {
  return lhs.shape() == rhs.shape() && lhs.data_type() == rhs.data_type();
}

}  // namespace xrt
}  // namespace oneflow

namespace std {
template <>
struct hash<oneflow::xrt::XrtShape> {
  size_t operator()(const oneflow::xrt::XrtShape &shape) const {
    return std::hash<std::string>()(shape.shape().ToString()) ^
           std::hash<int64_t>()(shape.data_type());
  }
};
}  // namespace std

#endif  // ONEFLOW_XRT_XRT_SHAPE_H_
