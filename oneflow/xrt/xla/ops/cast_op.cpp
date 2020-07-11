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
    DataType src_dtype = ctx->SoleInputType();
    xla::XlaOp in = ctx->SoleInput();
    if (src_dtype == dest_dtype) {
      ctx->SetSoleOutput(in);
    } else {
      xla::PrimitiveType data_type = DataTypeToPrimitiveType(dest_dtype);
      ctx->SetSoleOutput(xla::ConvertElementType(in, data_type));
    }
  }
};

REGISTER_XLA_OP_KERNEL(Cast, CastOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
