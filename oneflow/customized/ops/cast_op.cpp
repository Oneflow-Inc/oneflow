#include <oneflow/core/common/data_type.pb.h>
#include <oneflow/core/framework/user_op_conf.h>
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

Maybe<void> TensorDescInfer(user_op::InferContext* ctx) {
  const user_op::TensorDesc* input_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("input_tensor", 0);
  user_op::TensorDesc* output_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("output_tensor", 0);
  *output_tensor_desc = *input_tensor_desc;
  DataType* dtype = output_tensor_desc->mut_data_type();
  *dtype = ctx->GetAttr<DataType>("dtype");
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  SbpSignatureBuilder()
      .Split("input_tensor", 0, 0)
      .Split("output_tensor", 0, 0)
      .MakeSplitSignatureListBuilder(
          ctx->LogicalTensorDesc4InputArgNameAndIndex("input_tensor", 0).shape().NumAxes())
      .Build(ctx->sbp_sig_list());
  SbpSignatureBuilder()
      .PartialSum("input_tensor", 0)
      .PartialSum("output_tensor", 0)
      .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("cast")
    .Input("input_tensor")
    .Attr("dtype", UserOpAttrType::kAtDataType)
    .Output("output_tensor")
    .SetTensorDescInferFn(TensorDescInfer)
    .SetGetSbpFn(GetSbpSignatures);

REGISTER_USER_OP_GRAD("cast").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                        user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("input_tensor", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    const DataType& dtype = op.TensorDesc4ArgNameAndIndex("input_tensor", 0).data_type();
    user_op::UserOpConfWrapper cast_grad_op =
        builder.Op("cast")
            .Input("input_tensor", op.GetGradTensorWithOpOutput("output_tensor", 0))
            .Output("output_tensor")
            .Attr<DataType>("dtype", dtype)
            .Build();
    op.BindGradTensorWithOpInput(cast_grad_op.output("output_tensor", 0), "input_tensor", 0);
    AddOp(cast_grad_op);
  }
});

}  // namespace
}  // namespace oneflow