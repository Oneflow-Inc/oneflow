#include "oneflow/core/framework/framework.h"
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

namespace user_op {

namespace {

typedef std::function<Maybe<void>(user_op::InferContext* ctx)> TensorDescInferFn;
typedef std::function<void(const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp)>
    GenBackwardOpConfFn;

TensorDescInferFn MakeFwTensorDescInferFn() {
  return [](user_op::InferContext* ctx) -> Maybe<void> {
    const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
    const int32_t spatial_ndim = x_shape.NumAxes() - 2;
    std::vector<int32_t> padding_after = ctx->Attr<std::vector<int32_t>>("padding");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t>& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    
    CHECK_EQ_OR_RETURN(spatial_ndim, 2); // only support 4-D tensor now. 
    CHECK_EQ_OR_RETURN(kernel_size.size(), spatial_ndim);
    for (int32_t kernel : kernel_size) { CHECK_GT_OR_RETURN(kernel, 0); }
    CHECK_EQ_OR_RETURN(strides.size(), spatial_ndim);
    for (int32_t stride : strides) { CHECK_GT_OR_RETURN(stride, 0); }
    CHECK_EQ_OR_RETURN(dilation_rate.size(), spatial_ndim);
    for (int32_t dilation : dilation_rate) { CHECK_GT_OR_RETURN(dilation, 1); }

    std::vector<int64_t> dhw_shape(spatial_ndim);
    for (int32_t i = 0; i < spatial_ndim; ++i) {
      dhw_shape[i] = (x_shape.At(idx_offset + i) + 2*padding[i]
                      - dilation_rate[i] * (kernel_size[i] - 1) - 1)
                         / strides[i]
                     + 1;
    }

    DimVector y_shape(3);
    y_shape.at(0) = x_shape.At(0);
    y_shape.at(1) =
        x_shape.At(c_dim)
        * std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<int>());
    y_shape.at(2) = std::accumulate(dhw_shape.begin(), dhw_shape.end(), 1, std::multiplies<int>());

    user_op::TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("y", 0);
    *y_desc->mut_shape() = Shape(y_shape);
    return Maybe<void>::Ok();
  };
}

Maybe<void> SetFwDTypeFn(user_op::SbpContext* ctx) {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0); 
  return Maybe<void>::Ok();
}


Maybe<void> FwGetSbpFn(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
       .Split(user_op::OpArg("x", 0), 0)
       .Split(user_op::OpArg("y", 0), 0)
       .Build();
  
  ctx->NewBuilder()
       .Split(user_op::OpArg("x", 0), 1)
       .Split(user_op::OpArg("y", 0), 1)
       .Build();
  return Maybe<void>::Ok();
}

// Maybe<void> BwTensorDescInferFn(user_op::InferContext* ctx) {
//   *ctx->TensorDesc4ArgNameAndIndex("dx", 0) = *ctx->TensorDesc4ArgNameAndIndex("x", 0);
//   return Maybe<void>::Ok();
// }

// Maybe<void> BwGetSbpFn(user_op::SbpContext* ctx) {
//   ctx->NewBuilder()
//       .Split(user_op::OpArg("x", 0), 0)
//       .Split(user_op::OpArg("y", 0), 0)
//       .Split(user_op::OpArg("dy", 0), 0)
//       .Split(user_op::OpArg("dx", 0), 0)
//       .Build();
//   ctx->NewBuilder()
//       .Split(user_op::OpArg("x", 0), 1)
//       .Split(user_op::OpArg("y", 0), 1)
//       .Split(user_op::OpArg("dy", 0), 1)
//       .Split(user_op::OpArg("dx", 0), 1)
//       .Build();
//   }
//   return Maybe<void>::Ok();
// }

// GenBackwardOpConfFn MakeGenBackwardOpConfFn() {
//   return [](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
//     if (op.NeedGenGradTensor4OpInput("x", 0)) {
//       user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
//       user_op::UserOpConfWrapper grad_op =
//           builder.Op("unfold_grad")
//               .Input("x", op.input("x", 0))
//               .Input("y", op.output("y", 0))
//               .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
//               .Output("dx")
//               .Attr("padding", op.attr<std::vector<int32_t>>("padding"))
//               .Attr("kernel_size", op.attr<std::vector<int32_t>>("kernel_size"))
//               .Attr("strides", op.attr<std::vector<int32_t>>("strides"))
//               .Attr("dilation_rate", op.attr<std::vector<int32_t>>("dilation_rate"))
//               .Build();
//       op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
//       AddOp(grad_op);
//     }
//   };
// }

}  // namespace

REGISTER_USER_OP("unfold")
    .Input("x")
    .Output("y")
    .Attr<std::vector<int32_t>>("kernel_size")
    .Attr<std::vector<int32_t>>("padding")
    .Attr<std::vector<int32_t>>("strides")
    .Attr<std::vector<int32_t>>("dilation_rate")
    .SetTensorDescInferFn(MakeFwTensorDescInferFn())
    .SetGetSbpFn(FwGetSbpFn)
    .SetDataTypeInferFn(SetFwDTypeFn);

// REGISTER_USER_OP("unfold_grad")
//     .Input("x")
//     .Input("y")
//     .Input("dy")
//     .Output("dx")
//     .Attr<std::vector<int32_t>>("kernel_size")
//     .Attr<std::vector<int32_t>>("padding")
//     .Attr<std::vector<int32_t>>("strides")
//     .Attr<std::vector<int32_t>>("dilation_rate")
//     .SetTensorDescInferFn(BwTensorDescInferFn)
//     .SetGetSbpFn(BwGetSbpFn);

// REGISTER_USER_OP_GRAD("unfold").SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn());

}  // namespace user_op

}  // namespace oneflow