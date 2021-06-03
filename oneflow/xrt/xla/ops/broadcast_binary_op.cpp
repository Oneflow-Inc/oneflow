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
#include "oneflow/core/common/shape_view.h"
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"

#include "oneflow/xrt/xla/ops/binary_op.h"
#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

template<typename BinaryOp>
class BcastBinaryOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext* ctx) override {
    Shape shape_a = ctx->InputShape("x_0");
    Shape shape_b = ctx->InputShape("y_0");

    int axes = std::max(shape_a.NumAxes(), shape_b.NumAxes());
    shape_a = CreateLeftExtendedShape(ShapeView(shape_a), axes);
    shape_b = CreateLeftExtendedShape(ShapeView(shape_b), axes);

    xla::XlaOp a = Reshape(ctx->Input("x_0"), shape_a);
    xla::XlaOp b = Reshape(ctx->Input("y_0"), shape_b);
    ctx->SetOutput("z_0", BinaryOp()(a, b));
  }
};

REGISTER_XLA_OP_KERNEL(BcastAdd, BcastBinaryOp<op::Add>).Finalize();
REGISTER_XLA_OP_KERNEL(BcastMul, BcastBinaryOp<op::Mul>).Finalize();
REGISTER_XLA_OP_KERNEL(BcastDiv, BcastBinaryOp<op::Div>).Finalize();
REGISTER_XLA_OP_KERNEL(BcastMin, BcastBinaryOp<op::Min>).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
