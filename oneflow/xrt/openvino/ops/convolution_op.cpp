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

#include <ngraph/op/convolution.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class ConvolutionOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext* ctx) override {
    std::shared_ptr<ngraph::Node> input = ctx->Input("in_0");
    std::shared_ptr<ngraph::Node> weight = ctx->Weight("weight_0");

    ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
    const auto& pads = ctx->Attr<std::vector<int32_t>>("padding_before");
    std::vector<int32_t> stride_attr = ctx->Attr<std::vector<int32_t>>("strides");
    std::vector<size_t> stride;
    stride.assign(stride_attr.begin(), stride_attr.end());
    std::vector<int32_t> dilation_attr = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    std::vector<size_t> dilation;
    dilation.assign(dilation_attr.begin(), dilation_attr.end());

    std::shared_ptr<ngraph::Node> ngraph_node = std::make_shared<ngraph::op::v1::Convolution>(
        input, weight, ngraph::Strides(stride), ngraph::CoordinateDiff({pads[0], pads[1]}),
        ngraph::CoordinateDiff({pads[0], pads[1]}), ngraph::Strides(dilation), pad_type);
    ngraph_node->set_friendly_name(ctx->op_name().c_str());

    ctx->SetOutput("out_0", ngraph_node);
  }
};

REGISTER_OPENVINO_OP_KERNEL(Conv2D, ConvolutionOp).Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
