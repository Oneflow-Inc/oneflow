#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("split_like")
    .Input("in")
    .InputWithMinimum("like", 2)
    .OutputWithMinimum("out", 2)
    .Attr("axis", UserOpAttrType::kAtInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const int32_t axis = ctx->Attr<int32_t>("axis");
      const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      const DimVector& in_dim_vec = in_desc->shape().dim_vec();
      int64_t dim_sum = 0;
      for (size_t i = 0; i < ctx->outputs().size(); ++i) {
        const user_op::TensorDesc* like_i_desc = ctx->TensorDesc4ArgNameAndIndex("like", i);
        for (int64_t j = 0; j < like_i_desc->shape().NumAxes(); ++j) {
          if (j != axis) { CHECK_EQ_OR_RETURN(in_desc->shape().At(j), like_i_desc->shape().At(j)); }
        }
        CHECK_EQ_OR_RETURN(in_desc->data_type(), like_i_desc->data_type());
      }
      for (size_t i = 0; i < ctx->outputs().size(); ++i) {
        const user_op::TensorDesc* like_i_desc = ctx->TensorDesc4ArgNameAndIndex("like", i);
        dim_sum += like_i_desc->shape().At(axis);
        *ctx->TensorDesc4ArgNameAndIndex("out", i) = *like_i_desc;
      }
      CHECK_EQ_OR_RETURN(dim_sum, in_dim_vec.at(axis));
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& user_op_conf) {
      for (size_t i = 0; i < user_op_conf.input_size("like"); ++i) {
        user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", i);
        CHECK_NOTNULL(like_modifier);
        like_modifier->set_use_header_only(true);
        like_modifier->set_requires_grad(false);
      }
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      for (size_t i = 0; i < ctx->outputs().size(); ++i) {
        *ctx->BatchAxis4ArgNameAndIndex("out", i) = *ctx->BatchAxis4ArgNameAndIndex("like", i);
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int32_t axis = ctx->Attr<int32_t>("axis");
      const user_op::TensorDesc& in_0_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in_0_tensor.shape().NumAxes()) {
        if (i == axis) { continue; }
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      std::vector<user_op::OpArg> like_arg_vec;
      int32_t like_size = ctx->outputs().size();
      FOR_RANGE(int32_t, i, 0, like_size) { like_arg_vec.push_back(user_op::OpArg("like", i)); }
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .PartialSum(like_arg_vec)
          .PartialSum(ctx->outputs())
          .Build();
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .Broadcast(like_arg_vec)
          .PartialSum(ctx->outputs())
          .Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
