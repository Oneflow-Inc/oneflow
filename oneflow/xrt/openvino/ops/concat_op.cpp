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
#include "absl/strings/str_cat.h"
#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/openvino/ops/op_kernel.h"

#include <ngraph/op/concat.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class ConcatOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext* ctx) override {
    int num_inputs = ctx->num_inputs();
    CHECK_GE(num_inputs, 2) << "Concat needs 2 inputs at least.";
    Shape in_shape = ctx->InputShape("in_0");
    int64_t axis = ctx->Attr<int64_t>("axis");
    if (axis < 0) { axis += in_shape.NumAxes(); }
    CHECK_GE(axis, 0);
    CHECK_LT(axis, in_shape.NumAxes());

    std::vector<std::shared_ptr<ngraph::Node>> in(num_inputs);
    for (int i = 0; i < num_inputs; ++i) { in[i] = ctx->Input(absl::StrCat("in_", i)); }
    std::shared_ptr<ngraph::Node> ngraph_node = std::make_shared<ngraph::op::Concat>(in, axis);
    ngraph_node->set_friendly_name(ctx->op_name().c_str());
    ctx->SetOutput("out_0", ngraph_node);
  }
};

REGISTER_OPENVINO_OP_KERNEL(Concat, ConcatOp).EnableTrainPhase().Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
