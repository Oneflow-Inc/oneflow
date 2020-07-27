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

class AddOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    int num_inputs = ctx->num_inputs();
    CHECK_GT(num_inputs, 0);
    Shape shape = ctx->InputShape("in_0");
    xla::XlaOp sum = ctx->Input("in_0");

    for (int i = 1; i < num_inputs; ++i) {
      std::string name = absl::StrCat("in_", i);
      CHECK_EQ(shape, ctx->InputShape(name));
      sum = xla::Add(sum, ctx->Input(name));
    }

    ctx->SetSoleOutput(sum);
  }
};

REGISTER_XLA_OP_KERNEL(Add, AddOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
