#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class TransposeOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    const auto& perm = ctx->Attr<std::vector<int32_t>>("perm");
    Shape x_shape = ctx->InputShape("in");
    CHECK_EQ(perm.size(), x_shape.NumAxes());

    xla::XlaOp x = ctx->Input("in");
    if (IsIdentity(perm)) {
      ctx->SetOutput("out", x);
    } else {
      std::vector<long long> transposed_order(x_shape.NumAxes());
      for (int i = 0; i < x_shape.NumAxes(); ++i) { transposed_order[i] = perm[i]; }
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

REGISTER_XLA_OP_KERNEL(Transpose, TransposeOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
