#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_OPS_UNARY_OP_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_OPS_UNARY_OP_H_

#include "oneflow/xrt/of2xla/xla_data_type.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace mola {
namespace op {

#define OFXLA_DECLARE_UNARY_OP(op)                                    \
  struct op {                                                         \
    xla::XlaOp operator()(const xla::XlaOp &x) { return xla::op(x); } \
  };

OFXLA_DECLARE_UNARY_OP(Abs);
OFXLA_DECLARE_UNARY_OP(Logistic);
OFXLA_DECLARE_UNARY_OP(Tanh);

#undef OFXLA_DECLARE_UNARY_OP

}  // namespace op
}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_OPS_UNARY_OP_H_
