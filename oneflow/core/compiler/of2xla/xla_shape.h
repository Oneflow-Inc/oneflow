#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_SHAPE_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_SHAPE_H_

#include "tensorflow/compiler/xla/shape.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace mola {

Shape ShapeFromXlaShape(const xla::Shape &xla_shape);

xla::Shape XlaShapeFromShape(const Shape &shape);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_SHAPE_H_
