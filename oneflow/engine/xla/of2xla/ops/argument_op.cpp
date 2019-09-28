#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler.h"
#include "oneflow/engine/xla/of2xla/xla_op_context.h"

namespace oneflow {
namespace mla {

class XlaArgumentOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp in = ctx->Input("in");
    ctx->SetOutput("out", in);
  }
};

REGISTER_XLA_OP_COMPILER(XlaArgument, XlaArgumentOp);

}  // namespace mla
}  // namespace oneflow
