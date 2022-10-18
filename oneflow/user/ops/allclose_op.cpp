#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {
Maybe<void> AllCloseOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0);
  const user_op::TensorDesc& y_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("other", 0);
  CHECK_EQ_OR_RETURN(x_tensor.shape(), y_tensor.shape())
      << Error::RuntimeError()
      << "Inconsistent input tensor shape. Shape of input tensor is: " << x_tensor.shape()
      << ", shape of other tensor is: " << y_tensor.shape();
  FOR_RANGE(size_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("input", 0), i)
        .Split(user_op::OpArg("other", 0), i)
        .PartialSum(user_op::OpArg("out", 0))
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> AllCloseOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, Shape({}));
  return Maybe<void>::Ok();
}
Maybe<void> AllCloseOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
Maybe<void> AllCloseOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& other = ctx->InputTensorDesc("other", 0);

  CHECK_OR_RETURN(input.data_type() == other.data_type())
      << Error::RuntimeError() << "Expected both tensors to have same dtype, but found "
      << DataType_Name(input.data_type()) << ", and " << DataType_Name(other.data_type());
  ctx->SetOutputDType("out", 0, kBool);
  return Maybe<void>::Ok();
}
}  // namespace oneflow
