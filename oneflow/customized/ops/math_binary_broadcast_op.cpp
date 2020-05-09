#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

bool IsScalartTensor(Shape* shape) {
    return shape->NumAxes() == 1 && shape->At(0) == 1;
}

}  // namespace

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
            SbpSignatureBuilder().Broadcast("x").Split("y", i).Split("z", i).Build(
                ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
        }
        FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
            SbpSignatureBuilder()
                .Split("x", x_shape.NumAxes() - 1 - i)
                .Split("y", y_shape.NumAxes() - 1 - i)
                .Split(ctx->outputs(), y_shape.NumAxes() - 1 - i)
                .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
        }
    } else if (x_shape.NumAxes() > y_shape.NumAxes()) {
        FOR_RANGE(int64_t, i, 0, x_shape.NumAxes() - y_shape.NumAxes()) {
            SbpSignatureBuilder().Split("x", i).Broadcast("y").Split("z", i).Build(
                ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
        }
        FOR_RANGE(int64_t, i, 0, y_shape.NumAxes()) {
            SbpSignatureBuilder()
                .Split("x", x_shape.NumAxes() - 1 - i)
                .Split("y", y_shape.NumAxes() - 1 - i)
                .Split(ctx->outputs(), x_shape.NumAxes() - 1 - i)
                .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
        }
    } else {
        FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
            if (x_shape.At(i) == 1 && y_shape.At(i) == 1) { continue; }
            if (x_shape.At(i) == y_shape.At(i)) {
                SbpSignatureBuilder()
                    .Split(ctx->inputs(), i)
                    .Split(ctx->outputs(), i)
                    .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
            } else if (x_shape.At(i) == 1) {
                SbpSignatureBuilder()
                    .Broadcast("x")
                    .Split("y", i)
                    .Split(ctx->outputs(), i)
                    .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
            } else if (y_shape.At(i) == 1) {
                SbpSignatureBuilder()
                    .Split("x", i)
                    .Broadcast("y")
                    .Split(ctx->outputs(), i)
                    .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
            } else {
                UNIMPLEMENTED();
            }
        }
    }
    return Maybe<void>::Ok();
}

#define REGISTER_BINARYBROADCAST_USER_OP(op_name) \
    REGISTER_USER_OP(op_name) \
        .Input("x") \
        .Input("y") \
        .Output("z") \
        .SetTensorDescInferFn(InferBroadcastTensorDescFn) \
        .SetBatchAxisInferFn(InferBroadcastBatchAxis) \
        .SetGetSbpFn(GetBroadcastSbpSignature);

REGISTER_BINARYBROADCAST_USER_OP("broadcast_add")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_sub")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_mul")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_div")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_minimum")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_maximum")
REGISTER_BINARYBROADCAST_USER_OP("broadcast_floor_mod")


}  // namespace oneflow