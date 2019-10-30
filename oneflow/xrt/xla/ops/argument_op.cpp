#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class XlaArgumentOp : public OpKernel {
 public:
  void Compile(OpContext *ctx) override {
    xla::XlaOp in = ctx->Input("in");
    ctx->SetOutput("out", in);
  }
};

REGISTER_XLA_OP_COMPILER(XlaArgument, XlaArgumentOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
