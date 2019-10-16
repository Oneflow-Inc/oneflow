#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

#include "oneflow/xla/of2xla/xla_data_type.h"

namespace oneflow {
namespace mola {

class CastOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    DataType dest_dtype = ctx->GetAttr<DataType>("data_type");
    DataType src_dtype = ctx->InputType("in");
    xla::XlaOp in = ctx->Input("in");
    if (src_dtype == dest_dtype) {
      ctx->SetOutput("out", in);
    } else {
      xla::PrimitiveType data_type = DataTypeToPrimitiveType(dest_dtype);
      ctx->SetOutput("out", xla::ConvertElementType(in, data_type));
    }
  }
};

REGISTER_XLA_OP_COMPILER(Cast, CastOp);

}  // namespace mola
}  // namespace oneflow
