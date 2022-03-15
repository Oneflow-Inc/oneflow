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

#include <ngraph/op/relu.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class ReluOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext* ctx) override {
    std::shared_ptr<ngraph::Node> ngraph_node =
        std::make_shared<ngraph::op::Relu>(ctx->Input("x_0"));
    ngraph_node->set_friendly_name(ctx->op_name().c_str());
    ctx->SetOutput("y_0", ngraph_node);
  }
};

REGISTER_OPENVINO_OP_KERNEL(Relu, ReluOp).EnableTrainPhase().Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
