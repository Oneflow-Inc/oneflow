#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

namespace user_op {

void OpKernel::InferShape(KernelInferContext* ctx) const {
  InferContext* op_infer_ctx = ctx->MutOpInferContext();
  CHECK_NOTNULL(op_infer_ctx);
  ctx->GetOpInferFn()(op_infer_ctx);
  for (const auto& arg_pair : ctx->outputs()) {
    const Shape& shape = *op_infer_ctx->Shape4ArgNameAndIndex(arg_pair.first, arg_pair.second);
    ctx->MutShapeView4ArgNameAndIndex(arg_pair.first, arg_pair.second)->set_shape(shape);
  }
}

}  // namespace user_op

}  // namespace oneflow
