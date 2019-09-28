#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler.h"
#include "oneflow/engine/xla/of2xla/xla_op_context.h"

#include "oneflow/engine/xla/of2xla/ops/unary_op.h"

namespace oneflow {
namespace mla {

template <typename UnaryOp>
class ApplyUnaryOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    ctx->SetOutput("out", UnaryOp()(ctx->Input("in")));
  }
};

REGISTER_XLA_OP_COMPILER(Sigmoid, ApplyUnaryOp<op::Logistic>);
REGISTER_XLA_OP_COMPILER(Tanh, ApplyUnaryOp<op::Tanh>);

struct Gelu {
  xla::XlaOp operator()(const xla::XlaOp &x) {
    xla::XlaOp dot_5 = xla::ScalarLike(x, 0.5f);
    xla::XlaOp inv_sqrt2 = xla::ScalarLike(x, std::sqrt(0.5f));
    xla::XlaOp one = xla::ScalarLike(x, 1.f);
    // cdf = erf(sqrt(0.5) * x)
    xla::XlaOp cdf = xla::Erf(xla::Mul(inv_sqrt2, x));
    // return 0.5 * x * (1.0 + cdf)
    return xla::Mul(xla::Mul(dot_5, x), xla::Add(one, cdf));
  }
};

REGISTER_XLA_OP_COMPILER(Gelu, ApplyUnaryOp<Gelu>);

struct Identity {
  xla::XlaOp operator()(const xla::XlaOp &x) {
    return x;
  }
};

REGISTER_XLA_OP_COMPILER(Identity, ApplyUnaryOp<Identity>);

}  // namespace mla
}  // namespace oneflow
