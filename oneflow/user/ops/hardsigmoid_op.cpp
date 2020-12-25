#include "oneflow/core/framework/framework.h"

namespace oneflow{

namespace{

REGISTER_USER_OP("hardsigmoid")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void>{
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
        Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
        *out_shape = *in_shape; 
        *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
        return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext *ctx) -> Maybe<void>{
        *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
        return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void>{
        const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
        FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()){
            ctx->NewBuilder()
                 .Split(user_op::OpArg("in", 0), i)
                 .Split(user_op::OpArg("out", 0), i)
                 .Build();
        }
        return Maybe<void>::Ok();
    });


REGISTER_USER_OP("hardsigmoid_grad")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void>{
        const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
        const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
        Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
        CHECK(*dy_shape == *y_shape);
        *dx_shape = *dy_shape; 
        return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext *ctx) -> Maybe<void>{
        *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("y", 0);
        return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void>{
        const user_op::TensorDesc& y_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
        FOR_RANGE(int64_t, i, 0, y_tensor.shape().NumAxes()){
            ctx->NewBuilder()
                 .Split(user_op::OpArg("y", 0), i)
                 .Split(user_op::OpArg("dy", 0), i)
                 .Split(user_op::OpArg("dx", 0), i)
                 .Build();
        }
        return Maybe<void>::Ok();
    });


REGISTER_USER_OP_GRAD("hardsigmoid").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext *ctx){
    const auto hardsigmoid_grad_op_name = ctx->FwOp().op_name() + "_grad";
    ctx->DefineOp(hardsigmoid_grad_op_name, [&ctx](user_op::BackwardOpBuilder &builder){
        return builder.OpTypeName("hardsigmoid_grad")
            .InputBind("y", ctx->FwOp().output("out", 0))
            .InputBind("dy", ctx->FwOp().output_grad("out", 0))
            .Output("dx")
            .Build();
    });
    ctx->FwOp().InputGradBind(user_op::OpArg("in", 0), 
                              [&ctx, &hardsigmoid_grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(hardsigmoid_grad_op_name).output("dx", 0);
                              });
});

} // namespace 

} // namespace oneflow