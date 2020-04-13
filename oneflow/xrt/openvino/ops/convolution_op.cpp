#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/openvino/ops/op_kernel.h"

#include <ngraph/op/convolution.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class ConvolutionOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext *ctx) override {
    std::shared_ptr<ngraph::Node> input = ctx->Input("in");
    std::shared_ptr<ngraph::Node> weight = ctx->Input("weight");

    ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
    if (ctx->GetAttr<std::string>("padding") == "same") {
      pad_type = ngraph::op::PadType::SAME_LOWER;
    } else {
      pad_type = ngraph::op::PadType::VALID;
    }
    std::vector<int32_t> stride_attr = ctx->GetAttr<std::vector<int32_t>>("strides");
    std::vector<size_t> stride;
    stride.assign(stride_attr.begin(), stride_attr.end());
    std::vector<int32_t> dilation_attr = ctx->GetAttr<std::vector<int32_t>>("dilation_rate");
    std::vector<size_t> dilation;
    dilation.assign(dilation_attr.begin(), dilation_attr.end());

    std::shared_ptr<ngraph::Node> ngraph_node = std::make_shared<ngraph::op::v1::Convolution>(
        input, weight, ngraph::Strides(stride), ngraph::CoordinateDiff({0, 0}),
        ngraph::CoordinateDiff({0, 0}), ngraph::Strides(dilation), pad_type);
    ngraph_node->set_friendly_name(ctx->op_name().c_str());

    ctx->SetOutput("out", ngraph_node);
  }
};

REGISTER_OPENVINO_OP_KERNEL(Conv2D, ConvolutionOp).Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
