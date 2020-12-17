#include "oneflow/core/framework/framework.h"

namespace oneflow{

namespace user_op{

REGISTER_USER_OP("offset_to_ndindex")
    .Input("index")
    .Input("dims")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx)-> Maybe<void> {
        const TensorDesc* index = ctx->TensorDesc4ArgNameAndIndex("index", 0); 
        const TensorDesc* dims = ctx->TensorDesc4ArgNameAndIndex("dims", 0); 
        int64_t index_num_axes = index->shape().NumAxes(); 
        int64_t dims_num_axes = dims->shape().NumAxes(); 
        CHECK_GT_OR_RETURN(index_num_axes, 0); 
        CHECK_GT_OR_RETURN(dims_num_axes, 0); 
        user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
        *out->mut_shape() = dims->shape();
        *out->mut_data_type() = dims->data_type();  
        return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
        return user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis(ctx);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void>{
        ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
        return Maybe<void>::Ok();
    });

}

}