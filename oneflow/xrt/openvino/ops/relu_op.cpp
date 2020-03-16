#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/openvino/ops/op_kernel.h"

#include <ngraph/op/relu.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class ReluOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext *ctx) override {
    std::shared_ptr<ngraph::Node> ngraph_node =
        std::make_shared<ngraph::op::Relu>(ctx->Input("in"));
    ngraph_node->set_friendly_name(ctx->op_name().c_str());
    ctx->SetOutput("out", ngraph_node);
  }
};

REGISTER_OPENVINO_OP_KERNEL(Relu, ReluOp).EnableTrainPhase().Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
