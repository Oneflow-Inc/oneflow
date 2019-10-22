#include "oneflow/xrt/xla/op_context.h"
#include "oneflow/xrt/xla/ops/op_compiler.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class XlaArgumentOp : public OpCompiler {
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
