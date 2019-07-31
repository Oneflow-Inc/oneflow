#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_OPS_OPERATORS_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_OPS_OPERATORS_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace mola {
namespace op {

#define DECLARE_BINARY_OP(op)                           \
  struct op {                                           \
    xla::XlaOp operator()(xla::XlaOp a, xla::XlaOp b) { \
      return xla::op(a, b);                             \
    }                                                   \
  };

DECLARE_BINARY_OP(Add);
DECLARE_BINARY_OP(Mul);
DECLARE_BINARY_OP(Div);

#undef DECLARE_BINARY_OP

}  // namespace op
}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_OPS_OPERATORS_H_
