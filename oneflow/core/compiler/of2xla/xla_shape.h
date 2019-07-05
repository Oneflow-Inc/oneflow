#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_SHAPE_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_SHAPE_H_

#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.pb.h"

namespace oneflow {
namespace mola {

Shape XlaShapeToOfShape(const xla::Shape &xla_shape);

xla::Shape OfShapeToXlaShape(const Shape &shape, DataType dtype);

xla::Shape OfShapeToXlaShape(const Shape &shape, xla::PrimitiveType type);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_SHAPE_H_
