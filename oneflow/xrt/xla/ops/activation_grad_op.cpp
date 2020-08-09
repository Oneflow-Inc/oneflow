/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
    xla::XlaOp y = ctx->Input("y_0");
    xla::XlaOp dy = ctx->Input("dy_0");
    xla::XlaOp one = xla::ScalarLike(y, 1.f);
    // dx = dy * (1 - y * y)
    xla::XlaOp dx = dy * (one - (y * y));
    ctx->SetOutput("dx_0", dx);
  }
};
REGISTER_XLA_OP_KERNEL(TanhGrad, TanhGradOp).Finalize();

class GeluGradOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp x = ctx->Input("x_0");
    xla::XlaOp dy = ctx->Input("dy_0");
    xla::XlaOp dot_5 = xla::ScalarLike(x, 0.5f);
    xla::XlaOp inv_sqrt2 = xla::ScalarLike(x, std::sqrt(0.5f));
    xla::XlaOp one = xla::ScalarLike(x, 1.f);

    xla::XlaOp coef = xla::ScalarLike(x, std::sqrt(2.f / std::acos(-1.f)));
    // coef = 1 + erf(sqrt(0.5) * x) + x * coef * exp(-0.5 * x * x)
    coef = one + xla::Erf(inv_sqrt2 * x) + (x * coef * xla::Exp(xla::Neg(dot_5) * x * x));

    ctx->SetOutput("dx_0", dot_5 * coef * dy);
  }
};
REGISTER_XLA_OP_KERNEL(GeluGrad, GeluGradOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
