#include "oneflow/core/framework/framework.h"
namespace oneflow {

namespace user_op {

REGISTER_USER_OP("ravel_multi_index")
    .InputWithMinimum("multi_index", 1)
    .Input("dims")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const TensorDesc* multi_index_0 = ctx->TensorDesc4ArgNameAndIndex("multi_index", 0);
      const TensorDesc* dims = ctx->TensorDesc4ArgNameAndIndex("dims", 0);
      int64_t multi_index_num_axes = multi_index_0->shape().NumAxes();
      CHECK_GT_OR_RETURN(multi_index_num_axes, 0);
      int64_t dims_num_axes = dims->shape().NumAxes();
      CHECK_GT_OR_RETURN(dims_num_axes, 0);
      
      int64_t multi_index_elem_cnt = ctx->inputs().size()-1; 
      CHECK_EQ(dims->shape().At(0), multi_index_elem_cnt);  
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);    
      *out->mut_shape() = Shape({multi_index_0->shape().elem_cnt()});
      *out->mut_data_type() = dims->data_type();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      return user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis(ctx);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    });

} // namespace user_op

} // namespace oneflow