#include "oneflow/xrt/xla/op_context.h"
#include "oneflow/xrt/xla/ops/op_compiler.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/ops/binary_op.h"
#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

template <typename BinaryOp>
class ScalarBinaryOp : public OpCompiler {
 public:
  void Compile(OpContext *ctx) override {
    xla::XlaOp scalar = Scalar(ctx);
    xla::XlaOp in = ctx->Input("in");

    ctx->SetOutput("out", BinaryOp()(in, scalar));
  }

  xla::XlaOp Scalar(OpContext *ctx) const {
    xla::XlaBuilder *builder = ctx->builder();
    DataType data_type = ctx->InputType("in");
    std::string type = ctx->AttrTypeInOneof("scalar_operand");
    if (type == "int_operand") {
      int64_t value = ctx->GetAttr<int64_t>(type);
      return IntegerLiteral(builder, data_type, value);
    } else {
      double value = ctx->GetAttr<double>(type);
      return FloatLiteral(builder, data_type, value);
    }
  }
};

REGISTER_XLA_OP_COMPILER(ScalarAdd, ScalarBinaryOp<op::Add>).Finalize();
REGISTER_XLA_OP_COMPILER(ScalarMul, ScalarBinaryOp<op::Mul>).Finalize();
REGISTER_XLA_OP_COMPILER(ScalarDiv, ScalarBinaryOp<op::Div>).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
