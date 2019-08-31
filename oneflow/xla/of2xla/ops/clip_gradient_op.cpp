#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

#include "oneflow/xla/of2xla/xla_helpers.h"

namespace oneflow {
namespace mola {

class ClipGradientOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override;
};

void ClipGradientOp::Compile(XlaOpContext *ctx) {
  // TODO(hjchen2)
  ctx->SetOutput("out", ctx->Input("gradient"));
}

REGISTER_XLA_OP_COMPILER(ClipGradient, ClipGradientOp);

}  // namespace mola
}  // namespace oneflow
