#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_SHAPE_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_SHAPE_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace oneflow {
namespace mola {

Shape XlaShapeToOfShape(const xla::Shape &xla_shape);

xla::Shape OfShapeToXlaShape(const Shape &shape, DataType dtype);

xla::Shape OfShapeToXlaShape(const Shape &shape, xla::PrimitiveType type);

// Returns shape[start_dim:end_dim]
Shape SliceShape(const Shape &shape, size_t start_dim, size_t end_dim);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_SHAPE_H_
