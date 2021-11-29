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
#include <ngraph/op/interpolate.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class UpsampleNearestOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext* ctx) override {
    // upsample_nearest only supports NCHW
    const int32_t scale = ctx->Attr<int32_t>("scale");
    if (ctx->Attr<std::string>("data_format") != "channels_first") {
      LOG(FATAL) << "upsample_nearest only supports NCHW";
    }
    std::vector<int64_t> out_shape = {ctx->InputShape("in_0").At(2) * scale,
                                      ctx->InputShape("in_0").At(3) * scale};
    ngraph::op::InterpolateAttrs attrs;
    attrs.axes.insert(2);
    attrs.axes.insert(3);
    attrs.mode = "nearest";
    attrs.align_corners = 0;
    attrs.antialias = 0;
    attrs.pads_begin.push_back(0);
    attrs.pads_end.push_back(0);
    std::shared_ptr<ngraph::Node> input = ctx->Input("in_0");
    std::shared_ptr<ngraph::Node> out_shape_node =
        std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape({2}), out_shape);
    std::shared_ptr<ngraph::Node> ngraph_node =
        std::make_shared<ngraph::op::Interpolate>(input, out_shape_node, attrs);
    ngraph_node->set_friendly_name(ctx->op_name().c_str());
    ctx->SetOutput("out_0", ngraph_node);
  }
};

REGISTER_OPENVINO_OP_KERNEL(UpsampleNearest, UpsampleNearestOp).EnableTrainPhase().Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
