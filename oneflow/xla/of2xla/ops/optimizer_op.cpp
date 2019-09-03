#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

#include "oneflow/xla/of2xla/xla_helpers.h"

namespace oneflow {
namespace mola {

class AdamOptimizerOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override;
};

void AdamOptimizerOp::Compile(XlaOpContext *ctx) {
  // TODO(hjchen2)
  ctx->SetOutput("out", ctx->Input("weight") * ctx->Input("gradient"));
}

REGISTER_XLA_OP_COMPILER(AdamOptimizer, AdamOptimizerOp);

}  // namespace mola
}  // namespace oneflow
