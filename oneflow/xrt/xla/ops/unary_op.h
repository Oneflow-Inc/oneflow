#ifndef ONEFLOW_XRT_XLA_OPS_UNARY_OP_H_
#define ONEFLOW_XRT_XLA_OPS_UNARY_OP_H_

#include "oneflow/xrt/xla/xla_data_type.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
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
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_OPS_UNARY_OP_H_
