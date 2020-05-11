#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

Maybe<void> TensorDescInfer(user_op::InferContext* ctx) {
  const user_op::TensorDesc* input_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  user_op::TensorDesc* output_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  *output_tensor_desc = *input_tensor_desc;
  DataType* dtype = output_tensor_desc->mut_data_type();
  *dtype = ctx->GetAttr<DataType>("dtype");
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  SbpSignatureBuilder()
      .Split("in", 0, 0)
      .Split("out", 0, 0)
      .MakeSplitSignatureListBuilder(
          ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes())
      .Build(ctx->sbp_sig_list());
  SbpSignatureBuilder().PartialSum("in", 0).PartialSum("out", 0).Build(
      ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("cast")
    .Input("in")
    .Attr("dtype", UserOpAttrType::kAtDataType)
    .Output("out")
    .SetTensorDescInferFn(TensorDescInfer)
    .SetGetSbpFn(GetSbpSignatures);

REGISTER_USER_OP_GRAD("cast").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                        user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    const DataType& dtype = op.TensorDesc4ArgNameAndIndex("in", 0).data_type();
    user_op::UserOpConfWrapper cast_grad_op =
        builder.Op("cast")
            .Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Output("out")
            .Attr<DataType>("dtype", dtype)
            .Build();
    op.BindGradTensorWithOpInput(cast_grad_op.output("out", 0), "in", 0);
    AddOp(cast_grad_op);
  }
});

}  // namespace
}  // namespace oneflow