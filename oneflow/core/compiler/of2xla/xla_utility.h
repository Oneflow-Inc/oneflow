#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_UTILITY_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_UTILITY_H_

#include <string>
#include "tensorflow/compiler/xla/shape.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {
namespace mola {

#define NoneString    ""
#define NonePtr       nullptr
#define ISNULL(x)     NonePtr == (x)
#define NOTNULL(x)    NonePtr != (x)

std::string ExtractOpTypeAsString(const Operator &op);

Shape ShapeFromXlaShape(const xla::Shape &xla_shape);

xla::Shape XlaShapeFromShape(const Shape &shape);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_UTILITY_H_
