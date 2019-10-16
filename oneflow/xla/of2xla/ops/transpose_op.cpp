#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_context.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace mola {

class TransposeOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    auto perm = ctx->GetAttr<std::vector<int32_t>>("perm");
    Shape x_shape = ctx->InputShape("in");
    CHECK_EQ(perm.size(), x_shape.NumAxes());

    xla::XlaOp x = ctx->Input("in");
    if (IsIdentity(perm)) {
      ctx->SetOutput("out", x);
    } else {
      std::vector<long long> transposed_order(x_shape.NumAxes());
      for (int i = 0; i < x_shape.NumAxes(); ++i) {
        transposed_order[i] = perm[i];
      }
      ctx->SetOutput("out", xla::Transpose(x, transposed_order));
    }
  }

  bool IsIdentity(const std::vector<int32_t> &perm) const {
    bool is_identity = true;
    for (int i = 0; i < perm.size(); ++i) {
      if (i != perm[i]) {
        is_identity = false;
        break;
      }
    }
    return is_identity || (perm.size() <= 1);
  }
};

REGISTER_XLA_OP_COMPILER(Transpose, TransposeOp);

}  // namespace mola
}  // namespace oneflow
