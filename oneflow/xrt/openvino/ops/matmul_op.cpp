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
#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/openvino/ops/op_kernel.h"
#include <ngraph/op/matmul.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class MatMulOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext* ctx) override {
    Shape a_shape = ctx->InputShape("a_0");
    Shape b_shape = ctx->InputShape("b_0");
    CHECK_GE(a_shape.NumAxes(), 2);
    CHECK_EQ(a_shape.NumAxes(), b_shape.NumAxes());

    bool transpose_a = ctx->Attr<bool>("transpose_a");
    bool transpose_b = ctx->Attr<bool>("transpose_b");
    auto a = ctx->Input("a_0");
    auto b = ctx->Input("b_0");

    std::shared_ptr<ngraph::Node> ngraph_node =
        std::make_shared<ngraph::op::v0::MatMul>(a, b, transpose_a, transpose_b);
    ngraph_node->set_friendly_name(ctx->op_name().c_str());
    ctx->SetOutput("out_0", ngraph_node);
  }
};

REGISTER_OPENVINO_OP_KERNEL(MatMul, MatMulOp).EnableTrainPhase().Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
