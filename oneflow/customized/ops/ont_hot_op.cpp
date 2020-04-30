#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

REGISTER_USER_OP("one_hot")
    .Input("indices")
    .Output("out")
    .Attr("depth", UserOpAttrType::kAtInt64)
    .Attr("on_value", UserOpAttrType::kAtFloat)
    .Attr("off_value", UserOpAttrType::kAtFloat)
    .Attr("dtype", UserOpAttrType::kAtDataType)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
      const DataType indices_dtype = *ctx->Dtype4ArgNameAndIndex("indices", 0);
      CHECK_OR_RETURN(IsIndexDataType(indices_dtype));
      auto dtype = ctx->GetAttr<DataType>("dtype");
      const int64_t depth = ctx->GetAttr<int64_t>("depth");
      DimVector dim_vec = indices_shape->dim_vec();
      dim_vec.push_back(depth);
      *ctx->Shape4ArgNameAndIndex("out", 0) = Shape(dim_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = dtype;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
