#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/attrs/nn.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

namespace {

tvm::relay::Expr GetTransposeInputExpr(tvm::relay::Expr input) {
  auto transpose_attrs = tvm::make_node<tvm::relay::TransposeAttrs>(); 
  transpose_attrs->axes = tvm::Array<tvm::Integer>{1, 0};
  auto transpose_op = tvm::relay::Op::Get("transpose");
  return tvm::relay::CallNode::make(transpose_op, {input}, tvm::Attrs(transpose_attrs), {});
}

}

class MatmulOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    const Shape& a_shape = ctx->GetShape4InputName("a");
    const Shape& b_shape = ctx->GetShape4InputName("b");
    bool transpose_a = ctx->GetAttr<bool>("transpose_a");
    bool transpose_b = ctx->GetAttr<bool>("transpose_b");

    tvm::relay::Expr data_input  = ctx->GetExpr4InputName("a");
    tvm::relay::Expr weight_input  = ctx->GetExpr4InputName("b");
    if (a_shape.NumAxes() == 2 && b_shape.NumAxes() == 2) {
      if (transpose_a) { data_input = GetTransposeInputExpr(data_input); }
      // the original shape of b is tranposed compared with shape that tvm required
      if (!transpose_b) { weight_input = GetTransposeInputExpr(weight_input); }

      auto dense_attrs = tvm::make_node<tvm::relay::DenseAttrs>(); 
      int64_t units = transpose_b ? b_shape.At(0) : b_shape.At(1);
      dense_attrs->units = tvm::relay::IndexExpr(static_cast<int32_t>(units));

      auto dense_op = tvm::relay::Op::Get("nn.dense");
      auto dense_expr = tvm::relay::CallNode::make(
          dense_op, {data_input, weight_input}, tvm::Attrs(dense_attrs), {});
      ctx->set_op_expr(dense_expr);
    } else {
      //TODO(niuchong): support batch_matmul
    }
  }
};

REGISTER_TVM_OP_KERNEL(MatMul, MatmulOp).EnableTrainPhase().Finalize();

}
}
}
