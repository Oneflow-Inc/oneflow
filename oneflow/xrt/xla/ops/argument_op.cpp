#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class ArgumentOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    // xla::XlaOp value = ctx->Variable("value");
    // ctx->SetOutput("value", value);
  }
};

REGISTER_XLA_OP_KERNEL(Argument, ArgumentOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
