#include "oneflow/xrt/tvm/ops/op_kernel.h"
#include <tvm/relay/attrs/transform.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class TransposeOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("input_0"));

    const Shape& in_shape = ctx->GetShape4InputName("input_0");
    std::vector<int32_t> axes = ctx->Attr<std::vector<int32_t>>("perm");
    CHECK_EQ(axes.size(), in_shape.NumAxes());

    tvm::Array<tvm::Integer> tvm_axes;
    for (int32_t dim : axes) { tvm_axes.push_back(dim); }
    auto transpose_attrs = tvm::runtime::make_object<tvm::relay::TransposeAttrs>();
    transpose_attrs->axes = tvm_axes;

    auto op = tvm::relay::Op::Get("transpose");
    auto expr = tvm::relay::Call(op, node_inputs, tvm::Attrs(transpose_attrs), {});
    ctx->SetExpr4OutputName("output_0", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(Transpose, TransposeOp).Finalize();

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow
