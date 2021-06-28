# include "oneflow/core/framework/framework.h"

namespace oneflow{

REGISTER_USER_OP("triu")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("diagonal")
    .SetTensorDescInferFn([](user_op::InferContext* ctx)->Maybe<void>{
        const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0); 
        user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0); 
        CHECK_GE_OR_RETURN(in->shape().NumAxes(), 2); 
        *out->mut_shape() = in->shape(); 
        *out->mut_is_dynamic() = in->is_dynamic(); 
        return Maybe<void>::Ok(); 
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx)->Maybe<void>{
        const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0); 
        user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0); 
        *out->mut_data_type() = in->data_type();
        return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in.shape().NumAxes() - 2) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      ctx->NewBuilder()
        .PartialSum(user_op::OpArg("in", 0))
        .PartialSum(user_op::OpArg("out", 0))
        .Build();
      return Maybe<void>::Ok();
    });

}