#ifndef ONEFLOW_ENGINE_XLA_OF2XLA_OPS_BINARY_OP_H_
#define ONEFLOW_ENGINE_XLA_OF2XLA_OPS_BINARY_OP_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace mla {
namespace op {

#define OFXLA_DECLARE_BINARY_OP(op)                     \
  struct op {                                           \
    xla::XlaOp operator()(xla::XlaOp a, xla::XlaOp b) { \
      return xla::op(a, b);                             \
    }                                                   \
  };

OFXLA_DECLARE_BINARY_OP(Add);
OFXLA_DECLARE_BINARY_OP(Mul);
OFXLA_DECLARE_BINARY_OP(Div);

#undef OFXLA_DECLARE_BINARY_OP

}  // namespace op
}  // namespace mla
}  // namespace oneflow

#endif  // ONEFLOW_ENGINE_XLA_OF2XLA_OPS_BINARY_OP_H_
