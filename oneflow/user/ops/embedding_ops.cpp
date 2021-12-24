#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::InferLogicalTensorDesc(user_op::InferContext *ctx) {
      DimVector out_dim_vec = ctx->InputShape("ids", 0).dim_vec();
      const int64_t embedding_vec_size = ctx->Attr<int64_t>("embedding_size");
      out_dim_vec.push_back(embedding_vec_size);
      *ctx->OutputShape("embeddings", 0) = Shape(out_dim_vec);
      return Maybe<void>::Ok();
    }

/*static*/ Maybe<void> EmbeddingLookupPlaceholderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {return InferLogicalTensorDesc(ctx);}

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::GetSbp(user_op::SbpContext *ctx) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("ids", 0), 0)
          .Split(user_op::OpArg("embeddings", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    }

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::ModifyInputArg(const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper &conf) {
      user_op::InputArgModifier* ids = GetInputArgModifierFn("ids", 0);
      CHECK_OR_RETURN(ids != nullptr);
      ids->set_requires_grad(false);
      return Maybe<void>::Ok();
    }

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::InferDataType(user_op::InferContext *ctx) {
      *ctx->OutputDType("embeddings", 0) = ctx->Attr<DataType>("dtype");
      return Maybe<void>::Ok();
    }

/* static */ Maybe<void> SgdEmbeddingUpdatePlaceholderOp::InferLogicalTensorDesc(user_op::InferContext *ctx) {
      return Maybe<void>::Ok();
    }

/*static*/ Maybe<void> SgdEmbeddingUpdatePlaceholderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {return InferLogicalTensorDesc(ctx);}

/* static */ Maybe<void> SgdEmbeddingUpdatePlaceholderOp::GetSbp(user_op::SbpContext *ctx) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("ids", 0), 0)
          .Split(user_op::OpArg("embedding_diff", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    }

/* static */ Maybe<void> SgdEmbeddingUpdatePlaceholderOp::InferDataType(user_op::InferContext *ctx) { return Maybe<void>::Ok(); }


REGISTER_USER_OP_GRAD("embedding_lookup_placeholder")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_update");
      user_op::UserOpConfWrapper grad_op =
          builder.Op("sgd_embedding_update_placeholder")
              .Input("ids", op.input("ids", 0))
              .Input("embedding_diff", op.GetGradTensorWithOpOutput("embeddings", 0))
              .Attr<std::string>("name", op.attr<std::string>("name"))
              .Build();
      AddOp(grad_op);
      return Maybe<void>::Ok();
    });

}
