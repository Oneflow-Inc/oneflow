#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/xla_data_type.h"

namespace oneflow {
namespace xrt {
namespace mola {

class CastOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    DataType dest_dtype = ctx->Attr<DataType>("dtype");
    DataType src_dtype = ctx->InputType("in_0");
    xla::XlaOp in = ctx->Input("in_0");
    if (src_dtype == dest_dtype) {
      ctx->SetOutput("out_0", in);
    } else {
      xla::PrimitiveType data_type = DataTypeToPrimitiveType(dest_dtype);
      ctx->SetOutput("out_0", xla::ConvertElementType(in, data_type));
    }
  }
};

REGISTER_XLA_OP_KERNEL(Cast, CastOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
