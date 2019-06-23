#include "tensorflow/compiler/xla/shape.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/compiler/of2xla/xla_shape.h"

namespace oneflow {
namespace mola {

Shape ShapeFromXlaShape(const xla::Shape &xla_shape) {
  // TODO(hjchen2)
  Shape shape;
  return shape;
}

xla::Shape XlaShapeFromShape(const Shape &shape) {
  // TODO(hjchen2)
  xla::Shape xla_shape;
  return xla_shape;
}

}  // namespace mola
}  // namespace oneflow
