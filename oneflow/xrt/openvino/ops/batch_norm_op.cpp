#include "absl/strings/str_cat.h"
#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/openvino/ops/op_kernel.h"

#include <ngraph/op/batch_norm.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class NormalizationOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext *ctx) override {
    std::shared_ptr<ngraph::Node> input = ctx->Input("in");
    std::shared_ptr<ngraph::Node> gamma = ctx->Input("gamma");
    std::shared_ptr<ngraph::Node> beta = ctx->Input("beta");
    std::shared_ptr<ngraph::Node> moving_mean = ctx->Input("moving_mean");
    std::shared_ptr<ngraph::Node> moving_variance = ctx->Input("moving_variance");
    float epsilon = ctx->GetAttr<float>("epsilon");
    std::shared_ptr<ngraph::Node> ngraph_node = std::make_shared<ngraph::op::BatchNormInference>(
        input, gamma, beta, moving_mean, moving_variance, epsilon);
    ngraph_node->set_friendly_name(ctx->op_name().c_str());
    ctx->SetOutput("out", ngraph_node);
  }
};

REGISTER_OPENVINO_OP_KERNEL(Normalization, NormalizationOp).Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
