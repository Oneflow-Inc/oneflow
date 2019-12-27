#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/ops/binary_op.h"
#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

template<typename BinaryOp>
class ScalarBinaryOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp scalar = Scalar(ctx);
    xla::XlaOp in = ctx->Input("in");

    ctx->SetOutput("out", BinaryOp()(in, scalar));
  }

  xla::XlaOp Scalar(XlaOpContext *ctx) const {
    xla::XlaBuilder *builder = ctx->builder();
    DataType data_type = ctx->InputType("in");
    std::string type = ctx->GetOneofType("scalar_operand");
    if (type == "int_operand") {
      int64_t value = ctx->GetAttr<int64_t>(type);
      return IntegerLiteral(builder, data_type, value);
    } else {
      double value = ctx->GetAttr<double>(type);
      return FloatLiteral(builder, data_type, value);
    }
  }
};

REGISTER_XLA_OP_KERNEL(ScalarAdd, ScalarBinaryOp<op::Add>).Finalize();
REGISTER_XLA_OP_KERNEL(ScalarMul, ScalarBinaryOp<op::Mul>).Finalize();
REGISTER_XLA_OP_KERNEL(ScalarDiv, ScalarBinaryOp<op::Div>).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
