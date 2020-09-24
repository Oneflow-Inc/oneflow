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

namespace oneflow {
namespace xrt {
namespace mola {

class BiasAddOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("a_0");
    Shape bias_shape = ctx->InputShape("b_0");
    CHECK_GE(in_shape.NumAxes(), 2);
    CHECK_EQ(bias_shape.NumAxes(), 1);

    CHECK_EQ(ctx->InputType("a_0"), ctx->InputType("b_0"));

    xla::XlaOp in = ctx->Input("a_0");
    xla::XlaOp bias = ctx->Input("b_0");

    // Channel dim for NCHW data formart
    int channel_dim = 1;
    ctx->SetOutput("out_0", xla::Add(in, bias, {channel_dim}));
  }
};

REGISTER_XLA_OP_KERNEL(BiasAdd, BiasAddOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
