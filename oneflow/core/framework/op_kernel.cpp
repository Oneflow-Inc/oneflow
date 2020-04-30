#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

namespace user_op {

void OpKernel::InferShape(KernelInferContext* ctx) const { ctx->NaiveInferShape(); }

}  // namespace user_op

}  // namespace oneflow
