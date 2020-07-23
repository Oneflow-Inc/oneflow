#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/customized/ops/nn_util.h"

namespace oneflow {
namespace user_op {

REGISTER_USER_OP("same_padding")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("num_spatial_dims", UserOpAttrType::kAtInt32)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("kernel_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .Attr("dilation_rate", UserOpAttrType::kAtListInt32)
    .Attr("floating_constant_value", UserOpAttrType::kAtDouble)
    .Attr("integral_constant_value", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      *y_desc = *x_desc;
      const std::string padding = ctx->Attr<std::string>("padding");
      const std::string data_format = ctx->Attr<std::string>("data_format");
      const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
      const std::vector<int32_t> strides = ctx->Attr<std::vector<int32_t>>("strides");
      const std::vector<int32_t> dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
      const size_t idx_offset = IdxOffset(data_format);
      const int32_t num_spatial_dims = ctx->Attr<int32_t>("num_spatial_dims");
      DimVector y_dim_vec(x_desc->shape().dim_vec());
      for (int32_t i = 0; i < num_spatial_dims; ++i) {
        int32_t padding_small = 0;
        int32_t padding_large = 0;
        CalcSamePadding(x_desc->shape().At(idx_offset + i), kernel_size.at(i), dilation_rate.at(i),
                        strides.at(i), padding, &padding_small, &padding_large);
        y_dim_vec[idx_offset + i] =
            x_desc->shape().At(idx_offset + i) + padding_small + padding_large;
      }
      *y_desc->mut_shape() = Shape(y_dim_vec);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("same_padding_grad")
    .Input("x_like")
    .Input("dy")
    .Output("dx")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("num_spatial_dims", UserOpAttrType::kAtInt32)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("kernel_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .Attr("dilation_rate", UserOpAttrType::kAtListInt32)
    .Attr("floating_constant_value", UserOpAttrType::kAtDouble)
    .Attr("integral_constant_value", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->TensorDesc4ArgNameAndIndex("dx", 0) = *ctx->TensorDesc4ArgNameAndIndex("x_like", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x_like", 0), 0)
          .Split(user_op::OpArg("dy", 0), 0)
          .Split(user_op::OpArg("dx", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("same_padding")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        const std::string padding = op.attr<std::string>("padding");
        const int32_t num_spatial_dims = op.attr<int32_t>("num_spatial_dims");
        const std::string data_format = op.attr<std::string>("data_format");
        const std::vector<int32_t> kernel_size = op.attr<std::vector<int32_t>>("kernel_size");
        const std::vector<int32_t> strides = op.attr<std::vector<int32_t>>("strides");
        const std::vector<int32_t> dilation_rate = op.attr<std::vector<int32_t>>("dilation_rate");

        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("same_padding_grad")
                .Input("x_like", op.input("x", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Output("dx")
                .Attr<int32_t>("num_spatial_dims", num_spatial_dims)
                .Attr<std::string>("padding", padding)
                .Attr<std::string>("data_format", data_format)
                .Attr<std::vector<int32_t>>("kernel_size", kernel_size)
                .Attr<std::vector<int32_t>>("strides", strides)
                .Attr<std::vector<int32_t>>("dilation_rate", dilation_rate)
                .Attr("floating_constant_value", op.attr<double>("floating_constant_value"))
                .Attr("integral_constant_value", op.attr<int64_t>("integral_constant_value"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace user_op
}  // namespace oneflow
