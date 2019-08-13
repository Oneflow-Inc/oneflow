#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

#include "oneflow/xla/of2xla/xla_helpers.h"

namespace oneflow {
namespace mola {

class TanhGradOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp y = ctx->Input("y");
    xla::XlaOp dy = ctx->Input("dy");
    xla::XlaOp one = xla::ScalarLike(y, 1.0);
    // dx = dy * (1 - y * y)
    xla::XlaOp dx = dy * (one - (y * y));
    ctx->SetOutput("dx", dx);
  }
};
REGISTER_XLA_OP_COMPILER(TanhGrad, TanhGradOp);

class GeluGradOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp x = ctx->Input("x");
    xla::XlaOp dy = ctx->Input("dy");
    xla::XlaOp dot_5 = xla::ScalarLike(x, 0.5);
    xla::XlaOp one = xla::ScalarLike(x, 1.0);
    xla::XlaOp two = xla::ScalarLike(x, 2.0);

    xla::XlaOp inv_sqrt2 = xla::Sqrt(dot_5);
    xla::XlaOp coef = xla::Sqrt(two / xla::Acos(xla::Neg(one)));
    // coef = 1 + erf(sqrt(0.5) * x) + x * coef * exp(-0.5 * x * x)
    coef = one + xla::Erf(inv_sqrt2 * x) +
        (x * coef * xla::Exp(xla::Neg(dot_5) * x * x));

    ctx->SetOutput("dx", dot_5 * coef * dy);
  }
};
REGISTER_XLA_OP_COMPILER(GeluGrad, GeluGradOp);

}  // namespace mola
}  // namespace oneflow
