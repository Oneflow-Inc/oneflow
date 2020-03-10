#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("clip_by_value")
    .Input("in")
    .Attr("min", UserOpAttrType::kAtFloat)
    .Attr("max", UserOpAttrType::kAtFloat)
    .Attr("has_min", UserOpAttrType::kAtBool)
    .Attr("has_max", UserOpAttrType::kAtBool)
    .Output("out")
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* params_shape = ctx->Shape4ArgNameAndIndex("params", 0);
      Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
      int64_t index_ndims = indices_shape->At(indices_shape->NumAxes() - 1);
      OF_CHECK_LE(index_ndims, params_shape->NumAxes());
      DimVector out_shape_vec = indices_shape->dim_vec();
      out_shape_vec.pop_back();
      FOR_RANGE(int64_t, i, index_ndims, params_shape->NumAxes()) {
        out_shape_vec.push_back(params_shape->At(i));
      }
      *ctx->Shape4ArgNameAndIndex("out", 0) = Shape(out_shape_vec);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("params", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& params_desc =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("params", 0);
      const user_op::TensorDesc& indices_desc =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
      int64_t indices_num_axes = indices_desc.shape().NumAxes();
      FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
        SbpSignatureBuilder()
            .Broadcast("params", 0)
            .Split("indices", 0, i)
            .Split("out", 0, i)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      int64_t index_ndims = indices_desc.shape().At(indices_num_axes - 1);
      FOR_RANGE(int64_t, i, index_ndims, params_desc.shape().NumAxes()) {
        SbpSignatureBuilder()
            .Split("params", 0, i)
            .Broadcast("indices", 0)
            .Split("out", 0, i - index_ndims + indices_num_axes - 1)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
