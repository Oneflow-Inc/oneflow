#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ auto FusedGluOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
    return Maybe<void>::Ok();
}

/* static */ auto FusedGluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
    // obtain input shape
    const Shape& x_in_shape = ctx->InputShape("x", 0);
    const Shape& w_in_shape = ctx->InputShape("w", 0);
    const Shape& b_in_shape = ctx->InputShape("b", 0);

    // check existance of optional args
    bool is_split_mode = false;
    if (ctx->has_input("v", 0)) { CHECK_NOTNULL(ctx->InputTensorDesc("c", 0)); }
    if (ctx->has_input("c", 0)) { CHECK_NOTNULL(ctx->InputTensorDesc("v", 0)); }
    if (ctx->has_input("v", 0) && ctx->has_input("c", 0)) { is_split_mode = true; }

    // check dimensions of x, w and b
    CHECK_GE_OR_RETURN(x_in_shape.NumAxes(), 2);     // support dimension larger than 2
    CHECK_EQ_OR_RETURN(w_in_shape.NumAxes(), 2);
    CHECK_EQ_OR_RETURN(b_in_shape.NumAxes(), 1);

    // check input shapes of w and b
    size_t x_num_axes = x_in_shape.NumAxes();
    CHECK_EQ_OR_RETURN(w_in_shape.At(1), x_in_shape.At(x_num_axes-1))
        << "get " << w_in_shape.At(1) << " and " << x_in_shape.At(x_num_axes-1);
    CHECK_EQ_OR_RETURN(b_in_shape.At(0), w_in_shape.At(0))
        << "get " << b_in_shape.At(0) << " and " << w_in_shape.At(0);
    if (!is_split_mode)
        CHECK_EQ_OR_RETURN(w_in_shape.At(1) % 2, 0);
    
    // check both dimensions and input shapes of v and c (optional)
    if (is_split_mode) {
        const Shape& v_in_shape = ctx->InputShape("v", 0);
        const Shape& c_in_shape = ctx->InputShape("c", 0);

        CHECK_EQ_OR_RETURN(v_in_shape.NumAxes(), 2);
        CHECK_EQ_OR_RETURN(c_in_shape.NumAxes(), 1);

        CHECK_EQ_OR_RETURN(v_in_shape.At(0), w_in_shape.At(0))
            << "get " << v_in_shape.At(0) << " and " << w_in_shape.At(0);
        CHECK_EQ_OR_RETURN(v_in_shape.At(1), w_in_shape.At(1))
            << "get " << v_in_shape.At(1) << " and " << w_in_shape.At(1);
        CHECK_EQ_OR_RETURN(c_in_shape.At(0), b_in_shape.At(0))
            << "get " << c_in_shape.At(0) << " and " << b_in_shape.At(0);
    }

    // set shape of the output tensor y
    Shape y_out_shape = ctx->InputShape("x", 0);    // borrow from input shape
    size_t y_num_axes = x_num_axes;
    y_out_shape.Set(y_num_axes-2, x_in_shape.At(x_num_axes-2));
    if (is_split_mode)
        y_out_shape.Set(y_num_axes-1, w_in_shape.At(0));
    else
        y_out_shape.Set(y_num_axes-1, w_in_shape.At(0)/2);
    user_op::TensorDesc* y_tensor = ctx->MutOutputTensorDesc("y", 0);
    y_tensor->set_shape(y_out_shape);

    // set shape of the output tensors of both matmul_wx and matmul_vx
    Shape matmul_shape = ctx->InputShape("x", 0);   // borrow from input shape
    matmul_shape.Set(x_num_axes-2, x_in_shape.At(x_num_axes-2));
    matmul_shape.Set(x_num_axes-1, w_in_shape.At(0));
    user_op::TensorDesc* matmul_wx_tensor = ctx->MutOutputTensorDesc("matmul_wx", 0);
    matmul_wx_tensor->set_shape(matmul_shape);
    if (is_split_mode) {
        user_op::TensorDesc* matmul_vx_tensor = ctx->MutOutputTensorDesc("matmul_vx", 0);
        matmul_vx_tensor->set_shape(matmul_shape);
    }

    return Maybe<void>::Ok();
}

/* static */ auto FusedGluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
    return InferLogicalTensorDesc(ctx);
}

/* static */ auto FusedGluOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
    // obtain input data types
    DataType in_x_dtype = ctx->InputDType("x", 0);

    // check types of x, w and b
    CHECK_EQ_OR_RETURN(ctx->InputDType("w", 0), in_x_dtype);
    CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), in_x_dtype);

    // check types of v and c (optional)
    if (ctx->has_input("v", 0) && ctx->has_input("c", 0)) {
        CHECK_EQ_OR_RETURN(ctx->InputDType("v", 0), in_x_dtype);
        CHECK_EQ_OR_RETURN(ctx->InputDType("c", 0), in_x_dtype);
    }

    // set output data type
    ctx->SetOutputDType("y", 0, in_x_dtype);
    ctx->SetOutputDType("matmul_wx", 0, in_x_dtype);
    ctx->SetOutputDType("matmul_vx", 0, in_x_dtype);

    return Maybe<void>::Ok();
}

} // namespace oneflow