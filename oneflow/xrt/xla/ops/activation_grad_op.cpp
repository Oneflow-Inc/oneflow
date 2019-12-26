#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

class TanhGradOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp y = ctx->Input("y");
    xla::XlaOp dy = ctx->Input("dy");
    xla::XlaOp one = xla::ScalarLike(y, 1.f);
    // dx = dy * (1 - y * y)
    xla::XlaOp dx = dy * (one - (y * y));
    ctx->SetOutput("dx", dx);
  }
};
REGISTER_XLA_OP_KERNEL(TanhGrad, TanhGradOp).Finalize();

class GeluGradOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp x = ctx->Input("x");
    xla::XlaOp dy = ctx->Input("dy");
    xla::XlaOp dot_5 = xla::ScalarLike(x, 0.5f);
    xla::XlaOp inv_sqrt2 = xla::ScalarLike(x, std::sqrt(0.5f));
    xla::XlaOp one = xla::ScalarLike(x, 1.f);

    xla::XlaOp coef = xla::ScalarLike(x, std::sqrt(2.f / std::acos(-1.f)));
    // coef = 1 + erf(sqrt(0.5) * x) + x * coef * exp(-0.5 * x * x)
    coef = one + xla::Erf(inv_sqrt2 * x) + (x * coef * xla::Exp(xla::Neg(dot_5) * x * x));

    ctx->SetOutput("dx", dot_5 * coef * dy);
  }
};
REGISTER_XLA_OP_KERNEL(GeluGrad, GeluGradOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
