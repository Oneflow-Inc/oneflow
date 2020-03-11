#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("where")
    .Input("condition")
    .Input("x")
    .Input("y")
    .Output("out")
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

}  // namespace oneflow
