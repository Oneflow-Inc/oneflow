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
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/ops/binary_op.h"
#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

template<typename BinaryOp>
class ScalarBinaryOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext* ctx) override {
    xla::XlaOp scalar = Scalar(ctx);
    xla::XlaOp in = ctx->SoleInput();

    ctx->SetSoleOutput(BinaryOp()(in, scalar));
  }

  xla::XlaOp Scalar(XlaOpContext* ctx) const {
    xla::XlaBuilder* builder = ctx->builder();
    DataType data_type = ctx->SoleInputType();
    if (ctx->Attr<bool>("has_int_operand")) {
      int64_t value = ctx->Attr<int64_t>("int_operand");
      return IntegerLiteral(builder, data_type, value);
    } else if (ctx->Attr<bool>("has_float_operand")) {
      double value = ctx->Attr<double>("float_operand");
      return FloatLiteral(builder, data_type, value);
    }
  }
};

REGISTER_XLA_OP_KERNEL(ScalarAdd, ScalarBinaryOp<op::Add>).Finalize();
REGISTER_XLA_OP_KERNEL(ScalarMul, ScalarBinaryOp<op::Mul>).Finalize();
REGISTER_XLA_OP_KERNEL(ScalarDiv, ScalarBinaryOp<op::Div>).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
