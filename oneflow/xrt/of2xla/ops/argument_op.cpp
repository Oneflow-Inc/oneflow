#include "oneflow/xrt/of2xla/xla_op_compiler.h"
#include "oneflow/xrt/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xrt/of2xla/xla_op_context.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace mola {

class XlaArgumentOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp in = ctx->Input("in");
    ctx->SetOutput("out", in);
  }
};

REGISTER_XLA_OP_COMPILER(XlaArgument, XlaArgumentOp);

}  // namespace mola
}  // namespace oneflow
