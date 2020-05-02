#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

REGISTER_USER_OP("one_hot")
    .Input("indices")
    .Output("out")
    .Attr("depth", UserOpAttrType::kAtInt64)
    .Attr("floating_on_value", UserOpAttrType::kAtDouble)
    .Attr("integer_on_value", UserOpAttrType::kAtInt64)
    .Attr("floating_off_value", UserOpAttrType::kAtDouble)
    .Attr("integer_off_value", UserOpAttrType::kAtInt64)
    .Attr("dtype", UserOpAttrType::kAtDataType)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const int64_t depth = ctx->GetAttr<int64_t>("depth");
      CHECK_GT_OR_RETURN(depth, 0);
      const user_op::TensorDesc* indices_desc = ctx->TensorDesc4ArgNameAndIndex("indices", 0);
      CHECK_OR_RETURN(IsIndexDataType(indices_desc->data_type()));
      CHECK_GT_OR_RETURN(indices_desc->shape().NumAxes(), 0);
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *indices_desc;
      auto dtype = ctx->GetAttr<DataType>("dtype");
      *out_desc->mut_data_type() = dtype;
      DimVector dim_vec = indices_desc->shape().dim_vec();
      dim_vec.push_back(depth);
      *out_desc->mut_shape() = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& indices =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
      SbpSignatureBuilder()
          .Split("indices", 0, 0)
          .Split("out", 0, 0)
          .MakeSplitSignatureListBuilder(indices.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
