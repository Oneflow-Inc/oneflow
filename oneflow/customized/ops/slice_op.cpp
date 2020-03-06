#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/ops/slice_util.h"

namespace oneflow {

REGISTER_USER_OP("slice_v2")
    .Input("x")
    .Output("y")
    .Attr("begin", UserOpAttrType::kAtListInt64)
    .Attr("end", UserOpAttrType::kAtListInt64)
    .Attr("stride", UserOpAttrType::kAtListInt64)
    .Attr("has_begin", UserOpAttrType::kAtListInt64)
    .Attr("has_end", UserOpAttrType::kAtListInt64)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      auto begin_vec = ctx->GetAttr<std::vector<int64_t>>("begin");
      auto end_vec = ctx->GetAttr<std::vector<int64_t>>("end");
      auto stride_vec = ctx->GetAttr<std::vector<int64_t>>("stride");
      auto has_begin_vec = ctx->GetAttr<std::vector<int64_t>>("has_begin");
      auto has_end_vec = ctx->GetAttr<std::vector<int64_t>>("has_end");
      CHECK_EQ_OR_RETURN(in_shape->NumAxes(), begin_vec.size());
      CHECK_EQ_OR_RETURN(in_shape->NumAxes(), end_vec.size());
      CHECK_EQ_OR_RETURN(in_shape->NumAxes(), stride_vec.size());
      CHECK_EQ_OR_RETURN(begin_vec.size(), has_begin_vec.size());
      CHECK_EQ_OR_RETURN(end_vec.size(), has_end_vec.size());

      const SbpParallel& out_sbp = ctx->SbpParallel4ArgNameAndIndex("y", 0);
      if (ctx->parallel_ctx().parallel_num() != 1 && out_sbp.has_split_parallel()
          && out_sbp.split_parallel().axis() == 0) {
        CHECK_EQ_OR_RETURN(has_begin_vec[0], 0);
        CHECK_EQ_OR_RETURN(has_end_vec[0], 0);
        CHECK_EQ_OR_RETURN(stride_vec[0], 1);
      }

      DimVector dim_vec(in_shape->NumAxes());
      FOR_RANGE(size_t, i, 0, dim_vec.size()) {
        int64_t begin = has_begin_vec[i] ? RegulateSliceIndex(begin_vec[i], in_shape->At(i)) : 0;
        int64_t end =
            has_end_vec[i] ? RegulateSliceIndex(end_vec[i], in_shape->At(i)) : in_shape->At(i);
        int64_t stride = stride_vec[i];
        CHECK_NE_OR_RETURN(stride, 0) << "slice stride cannot be 0";
        if (stride > 0) {
          CHECK_LT_OR_RETURN(begin, end)
              << "If begin is not less than end when stride > 0, slice will output "
                 "empty result that it is not support";
        } else {
          CHECK_GT_OR_RETURN(begin, end)
              << "If begin is not more than end when stride < 0, slice will output "
                 "empty result that it is not support";
        }
        int64_t align = (begin > end) ? 1 : -1;
        dim_vec[i] = (end - begin + align) / stride + 1;
      }
      *ctx->Shape4ArgNameAndIndex("y", 0) = Shape(dim_vec);
      *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      int64_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes();
      auto stride_vec = ctx->GetAttr<std::vector<int64_t>>("stride");
      auto has_begin_vec = ctx->GetAttr<std::vector<int64_t>>("has_begin");
      auto has_end_vec = ctx->GetAttr<std::vector<int64_t>>("has_end");
      FOR_RANGE(int64_t, axis, 0, num_axes) {
        if (has_begin_vec[axis] == 0 && has_end_vec[axis] == 0 && stride_vec[axis] == 1) {
          SbpSignatureBuilder()
              .Split("x", 0, axis)
              .Split("y", 0, axis)
              .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
        }
      }
      SbpSignatureBuilder().PartialSum("x", 0).PartialSum("y", 0).Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("slice_grad_v2")
    .Input("dy")
    .Input("like")
    .Output("dx")
    .Attr("begin", UserOpAttrType::kAtListInt64)
    .Attr("end", UserOpAttrType::kAtListInt64)
    .Attr("stride", UserOpAttrType::kAtListInt64)
    .Attr("has_begin", UserOpAttrType::kAtListInt64)
    .Attr("has_end", UserOpAttrType::kAtListInt64)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
      auto begin_vec = ctx->GetAttr<std::vector<int64_t>>("begin");
      auto end_vec = ctx->GetAttr<std::vector<int64_t>>("end");
      auto stride_vec = ctx->GetAttr<std::vector<int64_t>>("stride");
      auto has_begin_vec = ctx->GetAttr<std::vector<int64_t>>("has_begin");
      auto has_end_vec = ctx->GetAttr<std::vector<int64_t>>("has_end");
      CHECK_EQ_OR_RETURN(like_shape->NumAxes(), begin_vec.size());
      CHECK_EQ_OR_RETURN(like_shape->NumAxes(), end_vec.size());
      CHECK_EQ_OR_RETURN(like_shape->NumAxes(), stride_vec.size());
      CHECK_EQ_OR_RETURN(begin_vec.size(), has_begin_vec.size());
      CHECK_EQ_OR_RETURN(end_vec.size(), has_end_vec.size());
      const SbpParallel& dx_sbp = ctx->SbpParallel4ArgNameAndIndex("dx", 0);
      if (ctx->parallel_ctx().parallel_num() != 1 && dx_sbp.has_split_parallel()
          && dx_sbp.split_parallel().axis() == 0) {
        CHECK_EQ_OR_RETURN(has_begin_vec[0], 0);
        CHECK_EQ_OR_RETURN(has_end_vec[0], 0);
        CHECK_EQ_OR_RETURN(stride_vec[0], 1);
      }
      *ctx->Shape4ArgNameAndIndex("dx", 0) = *like_shape;
      DataType* like_data_type = ctx->Dtype4ArgNameAndIndex("like", 0);
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("dy", 0), *like_data_type);
      *ctx->Dtype4ArgNameAndIndex("dx", 0) = *like_data_type;
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
      CHECK(like_arg_modifier != nullptr);
      like_arg_modifier->set_use_header_only(true);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      int64_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes();
      auto stride_vec = ctx->GetAttr<std::vector<int64_t>>("stride");
      auto has_begin_vec = ctx->GetAttr<std::vector<int64_t>>("has_begin");
      auto has_end_vec = ctx->GetAttr<std::vector<int64_t>>("has_end");
      FOR_RANGE(int64_t, axis, 0, num_axes) {
        if (has_begin_vec[axis] == 0 && has_end_vec[axis] == 0 && stride_vec[axis] == 1) {
          SbpSignatureBuilder()
              .Split("dy", 0, axis)
              .Split("like", 0, axis)
              .Split("dx", 0, axis)
              .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
        }
      }
      SbpSignatureBuilder()
          .PartialSum(ctx->inputs())
          .PartialSum(ctx->outputs())
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      SbpSignatureBuilder().PartialSum("dy", 0).Broadcast("like", 0).PartialSum("dx", 0).Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      SbpSignatureBuilder().Broadcast("dy", 0).PartialSum("like", 0).Broadcast("dx", 0).Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("slice_v2")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("slice_grad_v2")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("like", op.input("x", 0))
                .Attr("begin", op.attr<std::vector<int64_t>>("begin"))
                .Attr("end", op.attr<std::vector<int64_t>>("end"))
                .Attr("stride", op.attr<std::vector<int64_t>>("stride"))
                .Attr("has_begin", op.attr<std::vector<int64_t>>("has_begin"))
                .Attr("has_end", op.attr<std::vector<int64_t>>("has_end"))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
