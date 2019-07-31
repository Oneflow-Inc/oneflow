#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

#include "oneflow/xla/of2xla/xla_helpers.h"
#include "oneflow/xla/of2xla/ops/binary_op.h"

namespace oneflow {
namespace mola {

template <typename BinaryOp>
class ScalarBinaryOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp scalar = Scalar(ctx);
    xla::XlaOp in = ctx->Input("in");

    ctx->SetOutput("out", BinaryOp()(in, scalar));
  }

  xla::XlaOp Scalar(XlaOpContext *ctx) const {
    xla::XlaBuilder *builder = ctx->builder();
    DataType data_type = ctx->InputType("in");
    bool has_int_operand = ctx->HasAttr("int_operand");
    if (has_int_operand) {
      int64_t value = ctx->GetAttr<int64_t>("int_operand");
      return IntegerLiteral(builder, data_type, value);
    } else {
      double value = ctx->GetAttr<double>("float_operand");
      return FloatLiteral(builder, data_type, value);
    }
  }
};

REGISTER_XLA_OP_COMPILER(ScalarAdd, ScalarBinaryOp<op::Add>);
REGISTER_XLA_OP_COMPILER(ScalarMul, ScalarBinaryOp<op::Mul>);
REGISTER_XLA_OP_COMPILER(ScalarDiv, ScalarBinaryOp<op::Div>);

}  // namespace mola
}  // namespace oneflow
