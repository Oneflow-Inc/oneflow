#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

class IdentityOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    ctx->SetOutput("out", ctx->Input("in"));
  }
};

REGISTER_XLA_OP_COMPILER(Identity, IdentityOp);

}  // namespace mola
}  // namespace oneflow
