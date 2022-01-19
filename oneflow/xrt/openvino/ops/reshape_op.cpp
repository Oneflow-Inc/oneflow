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
#include <ngraph/op/constant.hpp>
#include <ngraph/op/reshape.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class ReshapeOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext* ctx) override {
    Shape in_shape = ctx->InputShape("in_0");
    Shape shape = ctx->OutputShape("out_0");
    CHECK_EQ(shape.Count(0), in_shape.Count(0));
    std::vector<size_t> dim_vec;
    for (int i = 0; i < shape.NumAxes(); ++i) { dim_vec.push_back(shape.At(i)); }

    std::shared_ptr<ngraph::Node> alpha_node = std::make_shared<ngraph::op::Constant>(
        ngraph::element::i32, ngraph::Shape({dim_vec.size()}), dim_vec);
    std::shared_ptr<ngraph::Node> ngraph_node =
        std::make_shared<ngraph::op::v1::Reshape>(ctx->Input("in_0"), alpha_node, false);
    ngraph_node->set_friendly_name(ctx->op_name().c_str());
    ctx->SetOutput("out_0", ngraph_node);
  }
};

REGISTER_OPENVINO_OP_KERNEL(Reshape, ReshapeOp).EnableTrainPhase().Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
