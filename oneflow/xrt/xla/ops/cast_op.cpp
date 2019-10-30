#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/xla_data_type.h"

namespace oneflow {
namespace xrt {
namespace mola {

class CastOp : public OpKernel {
 public:
  void Compile(OpContext *ctx) override {
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

REGISTER_XLA_OP_COMPILER(Cast, CastOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
