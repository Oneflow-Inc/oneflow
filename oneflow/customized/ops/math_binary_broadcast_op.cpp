#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

bool IsScalartTensor(Shape* shape) {
    return shape->NumAxes() == 1 && shape->At(0) == 1;
}

Maybe<void> InferBroadcastTensorDescFn(user_op::InferContext *ctx) {
    const user_op::TensorDesc* tensor_x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
    const user_op::TensorDesc* tensor_y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
    user_op::TensorDesc* tensor_z = ctx->TensorDesc4ArgNameAndIndex("z", 0);
    Shape* shape_x = ctx->Shape4ArgNameAndIndex("x", 0);
    Shape* shape_y = ctx->Shape4ArgNameAndIndex("y", 0);
    Shape* shape_z = ctx->Shape4ArgNameAndIndex("z", 0);
    size_t output_num_axes = std::max(shape_x->NumAxes(), shape_y->NumAxes());
    if (IsScalartTensor(shape_x)) {
        *shape_z = *shape_y;
    } else if (IsScalartTensor(shape_y)) {
        *shape_z = *shape_x;
    } else {
        const auto& x_shape = CreateLeftExtendedShape(ShapeView(*shape_x), output_num_axes);
        const auto& y_shape = CreateLeftExtendedShape(ShapeView(*shape_y), output_num_axes);
        *shape_z = *shape_x;
        Shape out_shape(x_shape);
        FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
            CHECK_OR_RETURN(x_shape.At(i) == 1 || y_shape.At(i) == 1 || x_shape.At(i) == y_shape.At(i));
            out_shape.Set(i, std::max(x_shape.At(i), y_shape.At(i)));
        }
        *shape_z = out_shape;
    }
    tensor_z->set_is_dynamic(tensor_x->is_dynamic() || tensor_y->is_dynamic());
    return Maybe<void>::Ok();
}

Maybe<void> InferBroadcastBatchAxis(user_op::BatchAxisContext* ctx) {
    // TODO (shijie)
    return Maybe<void>::Ok();
} 

Maybe<void> GetBroadcastSbpSignature(user_op::SbpContext* ctx) {
    const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
    const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
    if (x_shape.NumAxes() < y_shape.NumAxes()) {
        FOR_RANGE(int64_t, i, 0, y_shape.NumAxes() - x_shape.NumAxes()) {
            ctx->NewBuilder().Broadcast(user_op::OpArg("x", 0)).Split(user_op::OpArg("y", 0), i).Split(user_op::OpArg("z", 0), i).Build();
        }
        FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
            ctx->NewBuilder()
                .Split(user_op::OpArg("x", 0), x_shape.NumAxes() - 1 - i)
                .Split(user_op::OpArg("y", 0), y_shape.NumAxes() - 1 - i)
                .Split(ctx->outputs(), y_shape.NumAxes() - 1 - i)
                .Build();
        }
    } else if (x_shape.NumAxes() > y_shape.NumAxes()) {
        FOR_RANGE(int64_t, i, 0, x_shape.NumAxes() - y_shape.NumAxes()) {
            ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Broadcast(user_op::OpArg("y", 0)).Split(user_op::OpArg("z", 0), i).Build();
        }
        FOR_RANGE(int64_t, i, 0, y_shape.NumAxes()) {
            ctx->NewBuilder()
                .Split(user_op::OpArg("x", 0), x_shape.NumAxes() - 1 - i)
                .Split(user_op::OpArg("y", 0), y_shape.NumAxes() - 1 - i)
                .Split(ctx->outputs(), x_shape.NumAxes() - 1 - i)
                .Build();
        }
    } else {
        FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
            if (x_shape.At(i) == 1 && y_shape.At(i) == 1) { continue; }
            if (x_shape.At(i) == y_shape.At(i)) {
                ctx->NewBuilder()
                    .Split(ctx->inputs(), i)
                    .Split(ctx->outputs(), i)
                    .Build();
            } else if (x_shape.At(i) == 1) {
                ctx->NewBuilder()
                    .Broadcast(user_op::OpArg("x", 0))
                    .Split(user_op::OpArg("y", 0), i)
                    .Split(ctx->outputs(), i)
                    .Build();
            } else if (y_shape.At(i) == 1) {
                ctx->NewBuilder()
                    .Split(user_op::OpArg("x", 0), i)
                    .Broadcast(user_op::OpArg("y", 0))
                    .Split(ctx->outputs(), i)
                    .Build();
            } else {
                UNIMPLEMENTED();
            }
        }
    }
    return Maybe<void>::Ok();
}

Maybe<void> GetBroadcastGradSbpSignature(user_op::SbpContext* ctx) {
    const user_op::TensorDesc& tensor_dz = ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0);
    FOR_RANGE(int64_t, i, 0, tensor_dz.shape().NumAxes()) {
        ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
    }
    return Maybe<void>::Ok();
}

}  // namespace

#define REGISTER_BINARYBROADCAST_USER_OP(op_name)         \
    REGISTER_USER_OP(op_name)                             \
        .Input("x")                                       \
        .Input("y")                                       \
        .Output("z")                                      \
        .SetTensorDescInferFn(InferBroadcastTensorDescFn) \
        .SetBatchAxisInferFn(InferBroadcastBatchAxis)     \
        .SetGetSbpFn(GetBroadcastSbpSignature);

REGISTER_BINARYBROADCAST_USER_OP("broadcast_add")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_sub")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_mul")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_div")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_minimum")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_maximum")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_floor_mod")

#define REGISTER_BINARYBROADCAST_XGRAD_USER_OP(op_name)                         \
    REGISTER_USER_OP(op_name)                                                 \
        .Input("x")                                                           \
        .Input("y")                                                           \
        .Input("dz")                                                          \
        .Output("dx")                                                         \
        .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> { \
            Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);              \
            Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);            \
            *dx_shape = *x_shape;                                             \
            return Maybe<void>::Ok();                                         \
        })                                                                    \
        .SetBatchAxisInferFn(InferBroadcastBatchAxis)                         \
        .SetGetSbpFn(GetBroadcastGradSbpSignature);

REGISTER_BINARYBROADCAST_XGRAD_USER_OP("broadcast_add_x_grad")
REGISTER_BINARYBROADCAST_XGRAD_USER_OP("broadcast_sub_x_grad")
REGISTER_BINARYBROADCAST_XGRAD_USER_OP("broadcast_mul_x_grad")
REGISTER_BINARYBROADCAST_XGRAD_USER_OP("broadcast_div_x_grad")

#define REGISTER_BINARYBROADCAST_YGRAD_USER_OP(op_name)                         \
    REGISTER_USER_OP(op_name)                                                 \
        .Input("x")                                                           \
        .Input("y")                                                           \
        .Input("dz")                                                          \
        .Output("dy")                                                         \
        .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> { \
            Shape* y_shape = ctx->Shape4ArgNameAndIndex("x", 0);              \
            Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);            \
            *dy_shape = *y_shape;                                             \
            return Maybe<void>::Ok();                                         \
        })                                                                    \
        .SetBatchAxisInferFn(InferBroadcastBatchAxis)                         \
        .SetGetSbpFn(GetBroadcastGradSbpSignature);

REGISTER_BINARYBROADCAST_YGRAD_USER_OP("broadcast_add_y_grad")
REGISTER_BINARYBROADCAST_YGRAD_USER_OP("broadcast_sub_y_grad")
REGISTER_BINARYBROADCAST_YGRAD_USER_OP("broadcast_mul_y_grad")
REGISTER_BINARYBROADCAST_YGRAD_USER_OP("broadcast_div_y_grad")

#define REGISTER_BINARYBROADCAST_USER_OP_GRAD(op_name_)                                         \
    REGISTER_USER_OP_GRAD(op_name_).SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, \
                                                             user_op::AddOpFn AddOp) {         \
        if (op.NeedGenGradTensor4OpInput("x", 0)) {                                            \
            user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_x_grad");               \
            user_op::UserOpConfWrapper in_grad_op =                                            \
                builder.Op(op_name_)                                                            \
                    .Input("x", op.input("x", 0))                                              \
                    .Input("y", op.input("y", 0))                                              \
                    .Input("dz", op.GetGradTensorWithOpOutput("z", 0))                         \
                    .Output("dx")                                                              \
                    .Build();                                                                  \
            op.BindGradTensorWithOpInput(in_grad_op.output("dx", 0), "x", 0);                  \
            AddOp(in_grad_op);                                                                 \
        }                                                                                      \
        if (op.NeedGenGradTensor4OpInput("y", 0)) {                                            \
            user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_y_grad");               \
            user_op::UserOpConfWrapper in_grad_op =                                            \
                builder.Op(op_name_)                                                            \
                    .Input("x", op.input("x", 0))                                              \
                    .Input("y", op.input("y", 0))                                              \
                    .Input("dz", op.GetGradTensorWithOpOutput("z", 0))                         \
                    .Output("dy")                                                              \
                    .Build();                                                                  \
            op.BindGradTensorWithOpInput(in_grad_op.output("dy", 0), "y", 0);                  \
            AddOp(in_grad_op);                                                                 \
        }                                                                                      \
    });

REGISTER_BINARYBROADCAST_USER_OP_GRAD("broadcast_add")
REGISTER_BINARYBROADCAST_USER_OP_GRAD("broadcast_sub")
REGISTER_BINARYBROADCAST_USER_OP_GRAD("broadcast_mul")
REGISTER_BINARYBROADCAST_USER_OP_GRAD("broadcast_div")

}  // namespace oneflow