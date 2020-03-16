#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/openvino/ops/op_kernel.h"
#include "absl/strings/str_cat.h"
#include <ngraph/op/add.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

class BiasAddOp : public OpenvinoOpKernel {
 public:
  void Compile(OpenvinoOpContext *ctx) override {
    int num_inputs = ctx->num_inputs();
    CHECK_GE(num_inputs, 2) << "ElementWiseOp needs 2 inputs at least.";

    Shape in_shape = ctx->InputShape("in_0");
    std::shared_ptr<ngraph::Node> result = ctx->Input("in_0");
    for (int i = 1; i < num_inputs; ++i) {
      std::string name = absl::StrCat("in_", i);
      CHECK_EQ(in_shape, ctx->InputShape(name));
      std::shared_ptr<ngraph::Node> result =
          std::make_shared<ngraph::op::v1::Add>(ctx->Input(name), result);
      result->set_friendly_name(absl::StrCat(ctx->op_name().c_str(), i));
      ;
    }
    ctx->SetOutput("out", result);
  }
};

// REGISTER_OPENVINO_OP_KERNEL(BiasAdd, BiasAddOp)
//    .EnableTrainPhase()
//    .Finalize();

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
