#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class AddOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    int num_inputs = ctx->num_inputs();
    CHECK_GT(num_inputs, 0);
    Shape shape = ctx->InputShape("in_0");
    xla::XlaOp sum = ctx->Input("in_0");

    for (int i = 1; i < num_inputs; ++i) {
      std::string name = absl::StrCat("in_", i);
      CHECK_EQ(shape, ctx->InputShape(name));
      sum = xla::Add(sum, ctx->Input(name));
    }

    ctx->SetOutput("out", sum);
  }
};

REGISTER_XLA_OP_KERNEL(Add, AddOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
