#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.pb.h"

namespace oneflow {

#define REGISTER_BROADCAST_BINARY_USER_OP(op_type_name)                                             \
  REGISTER_USER_OP(op_type_name)                                                                    \
      .Input("a")                                                                                   \
      .Input("b")                                                                                   \
      .Output("out")                                                                                \
      .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {                         \
        Shape* a_tensor = ctx->TensorDesc4ArgNameAndIndex("a", 0);                                  \
        Shape* b_tensor = ctx->TensorDesc4ArgNameAndIndex("b", 0);                                  \
        Shape* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);                              \
  CHECK_EQ_OR_RETURN(a_tensor->data_type(), b_tensor->data_type());                                 \
  size_t output_num_axes = std::max(a_tensor->shape().NumAxes(), b_tensor->shape().NumAxes());      \
  if (IsScalarBlob(a_tensor)) {                                                                     \
    *out_tensor = *b_tensor;                                                                        \
  } else if (IsScalarBlob(b_tensor)) {                                                              \
    *out_tensor = *a_tensor;                                                                        \
  } else {                                                                                          \
    const auto& a_shape = CreateLeftExtendedShape(ShapeView(a_tensor->shape()), output_num_axes);   \
    const auto& b_shape = CreateLeftExtendedShape(ShapeView(b_tensor->shape()), output_num_axes);   \
    *out_tensor = *a_tensor;                                                                        \
    Shape out_shape(a_tensor->shape());                                                             \
    FOR_RANGE(int64_t, i, 0, a_tensor->shape().NumAxes()) {                                         \
      CHECK_OR_RETURN(a_tensor->shape().At(i) == 1 || b_tensor->shape().At(i) == 1 || a_tensor->shape().At(i) == b_tensor->shape().At(i));\
      out_shape().Set(i, std::max(a_tensor->shape().At(i), b_tensor->shape().At(i)));               \
    }                                                                                               \
    out_tensor->mut_shape() = out_shape;                                                            \
  }                                                                                                 \
  out_tensor->set_is_dynamic(a_tensor->is_dynamic() || b_tensor->is_dynamic());                     \
        return Maybe<void>::Ok();                                                                   \
      })                                                                                            \
      .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {                                    \
  const Shape& a_shape = Shape4ArgNameAndIndex("a")->shape();                                       \
  const Shape& b_shape = Shape4ArgNameAndIndex("b")->shape();                                       \
  if (a_shape.NumAxes() < b_shape.NumAxes()) {                                                      \
    FOR_RANGE(int64_t, i, 0, b_shape.NumAxes() - a_shape.NumAxes()) {                               \
      ctx->NewBuilder()                                                                             \
        .Broadcast(user_op::OpArg("a", 0))                                                          \
        .Split(user_op::OpArg("b", 0), i)                                                           \
        .Split(user_op::OpArg("out", 0))                                                            \
        .Build();                                                                                   \
    }                                                                                               \
    FOR_RANGE(int64_t, i, 0, a_shape.NumAxes()) {                                                   \
      ctx->NewBuilder()                                                                             \
          .Split(user_op::OpArg("a", 0), i, a_shape.NumAxes() - 1 - i)                              \
          .Split(user_op::OpArg("b", 0), i, b_shape.NumAxes() - 1 - i)                              \
          .Split(user_op::OpArg("out", 0), b_shape.NumAxes() - 1 - i)                               \
          .Build();                                                                                 \
    }                                                                                               \
  } else if (a_shape.NumAxes() > b_shape.NumAxes()) {                                               \
    FOR_RANGE(int64_t, i, 0, a_shape.NumAxes() - b_shape.NumAxes()) {                               \
      ctx->NewBuilder()                                                                             \
        .Split(user_op::OpArg("a", 0), i)                                                           \
        .Broadcast(user_op::OpArg("b", 0))                                                          \
        .Split(user_op::OpArg("out", 0), i)                                                         \
        .Build();                                                                                   \
    }                                                                                               \
    FOR_RANGE(int64_t, i, 0, b_shape.NumAxes()) {                                                   \
      ctx->NewBuilder()                                                                             \
          .Split(user_op::OpArg("a", 0), a_shape.NumAxes() - 1 - i)                                 \
          .Split(user_op::OpArg("b", 0), b_shape.NumAxes() - 1 - i)                                 \
          .Split(user_op::OpArg("out", 0), a_shape.NumAxes() - 1 - i)                               \
          .Build();                                                                                 \
    }                                                                                               \
  } else {                                                                                          \
    FOR_RANGE(int64_t, i, 0, a_shape.NumAxes()) {                                                   \
      if (a_shape.At(i) == 1 && b_shape.At(i) == 1) { continue; }                                   \
      if (a_shape.At(i) == b_shape.At(i)) {                                                         \
        ctx->NewBuilder()                                                                           \
            .Split(input_bns(), i)                                                                  \
            .Split(user_op::OpArg("out", 0), i)                                                     \
            .Build();                                                                               \
      } else if (a_shape.At(i) == 1) {                                                              \
        ctx->NewBuilder()                                                                           \
            .Broadcast(user_op::OpArg("a", 0))                                                      \
            .Split(user_op::OpArg("b", 0), i)                                                       \
            .Split(user_op::OpArg("out", 0), i)                                                     \
            .Build();                                                                               \
      } else if (b_shape.At(i) == 1) {                                                              \
        ctx->NewBuilder()                                                                           \
            .Split(user_op::OpArg("a", 0), i)                                                       \
            .Broadcast(user_op::OpArg("b", 0))                                                      \
            .Split(user_op::OpArg("out", 0), i)                                                     \
            .Build();                                                                               \
      } else {                                                                                      \
        UNIMPLEMENTED();                                                                            \
      }                                                                                             \
    }                                                                                               \
  }                                                                                                 \
        return Maybe<void>::Ok();                                                                   \
      });

REGISTER_BROADCAST_BINARY_USER_OP("broadcast_greater");
REGISTER_BROADCAST_BINARY_USER_OP("broadcast_greater_equal");
REGISTER_BROADCAST_BINARY_USER_OP("broadcast_less");
REGISTER_BROADCAST_BINARY_USER_OP("broadcast_less_equal");
REGISTER_BROADCAST_BINARY_USER_OP("broadcast_equal");
REGISTER_BROADCAST_BINARY_USER_OP("broadcast_not_equal");
REGISTER_BROADCAST_BINARY_USER_OP("broadcast_logical_and");

// add, mul, div, sub register op on their own, kernel user func ptr

REGISTER_USER_OP("broadcast_binary_x_grad")
    .Input("x")
    .Input("y")
    .Input("dz")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* dz_shape = ctx->Shape4ArgNameAndIndex("dz", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      CHECK((*y_shape == *x_shape) && (*dz_shape == *x_shape));
      *dx_shape = *x_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("broadcast_binary_y_grad")
    .Input("x")
    .Input("y")
    .Input("dz")
    .Output("dy")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* dz_shape = ctx->Shape4ArgNameAndIndex("dz", 0);
      Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      CHECK((*y_shape == *x_shape) && (*dz_shape == *x_shape));
      *dy_shape = *y_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

#define REGISTER_BROADCAST_BINATY_USER_OP_GRAD(op_type_name)                                       \
  REGISTER_USER_OP_GRAD(op_type_name)                                                              \
      .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {       \
        if (op.NeedGenGradTensor4OpInput("x", 0)) {                                                \
          user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_x_grad");                     \
          user_op::UserOpConfWrapper binary_grad_op =                                              \
              builder.Op("broadcast_binary_x_grad")                                                \
                  .Input("x", op.input("x", 0))                                                    \
                  .Input("y", op.input("y", 0))                                                    \
                  .Input("dz", op.GetGradTensorWithOpOutput("z", 0))                               \
                  .Output("dx")                                                                    \
                  .Build();                                                                        \
          op.BindGradTensorW - ithOpInput(binary_grad_op.output("dx", 0), "x", 0);                 \
          AddOp(binary_grad_op);                                                                   \
        }                                                                                          \
        if (op.NeedGenGradTensor4OpInput("y", 0)) {                                                \
          user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_y_grad");                     \
          user_op::UserOpConfWrapper binary_grad_op =                                              \
              builder.Op("broadcast_binary_y_grad")                                                \
                  .Input("x", op.input("x", 0))                                                    \
                  .Input("y", op.input("y", 0))                                                    \
                  .Input("dz", op.GetGradTensorWithOpOutput("z", 0))                               \
                  .Output("dy")                                                                    \
                  .Build();                                                                        \
          op.BindGradTensorWithOpInput(binary_grad_op.output("dy", 0), "y", 0);                    \
          AddOp(binary_grad_op);                                                                   \
        }                                                                                          \
      });
REGISTER_BROADCAST_BINARY_USR_OP_("broadcast_greater");
REGISTER_BROADCAST_BINARY_USR_OP("broadcast_greater_equal");
REGISTER_BROADCAST_BINARY_USR_OP("broadcast_less");
REGISTER_BROADCAST_BINARY_USR_OP("broadcast_less_equal");
REGISTER_BROADCAST_BINARY_USR_OP("broadcast_equal");
REGISTER_BROADCAST_BINARY_USR_OP("broadcast_not_equal");
REGISTER_BROADCAST_BINARY_USR_OP("broadcast_logical_and");

}  // namespace oneflow
