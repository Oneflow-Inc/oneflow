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
#include "absl/strings/str_cat.h"
#include <ngraph/op/add.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class AddOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext* ctx) override {
    int num_inputs = ctx->num_inputs();
    CHECK_GE(num_inputs, 2) << "ElementWiseOp needs 2 inputs at least.";
    Shape in_shape = ctx->InputShape("in_0");
    std::shared_ptr<ngraph::Node> result = ctx->Input("in_0");
    for (int i = 1; i < num_inputs; ++i) {
      std::string name = absl::StrCat("in_", i);
      CHECK_EQ(in_shape, ctx->InputShape(name));
      result = std::make_shared<ngraph::op::v1::Add>(ctx->Input(name), result);
      result->set_friendly_name(absl::StrCat(ctx->op_name().c_str(), i));
    }
    ctx->SetOutput("out_0", result);
  }
};

REGISTER_OPENVINO_OP_KERNEL(Add, AddOp).EnableTrainPhase().Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
