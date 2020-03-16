#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/openvino/ops/op_kernel.h"
#include <ngraph/op/constant.hpp>
#include <ngraph/op/fused/prelu.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class LeakyReluOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext *ctx) override {
    float alpha = ctx->GetAttr<float>("alpha");
    std::shared_ptr<ngraph::Node> input = ctx->Input("in");
    std::shared_ptr<ngraph::Node> alpha_node =
        std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape({1}), &alpha);
    std::shared_ptr<ngraph::Node> ngraph_node =
        std::make_shared<ngraph::op::PRelu>(input, alpha_node);
    ngraph_node->set_friendly_name(ctx->op_name().c_str());
    ctx->SetOutput("out", ngraph_node);
  }
};

REGISTER_OPENVINO_OP_KERNEL(LeakyRelu, LeakyReluOp).EnableTrainPhase().Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
