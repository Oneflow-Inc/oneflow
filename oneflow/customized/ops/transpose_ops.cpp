#include <oneflow/core/common/maybe.h>
#include <oneflow/core/framework/op_registration.h>
#include <oneflow/core/framework/tensor_desc.h>
#include "oneflow/core/framework/framework.h"

namespace oneflow {

void CheckIsPerm(const std::vector<int32_t>& perm) {
  std::vector<bool> is_used(perm.size(), false);
  FOR_RANGE(size_t, i, 0, perm.size()) {
    CHECK_GE(perm[i], 0);
    CHECK_LE(perm[i], perm.size());
    CHECK_EQ(is_used[perm[i]], false);
    is_used[perm[i]] = true;
  }
}

REGISTER_USER_OP("transpose")
    .Input("input")
    .Output("output")
    .Attr("perm", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("input", 0);
      user_op::TensorDesc* out_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("output", 0);
      const Shape& in_shape = in_tensor_desc->shape();
      Shape* out_shape = out_tensor_desc->mut_shape();
      const auto& perm = ctx->GetAttr<std::vector<int32_t>>("perm");
      CHECK_EQ_OR_RETURN(perm.size(), in_shape.NumAxes());
      CheckIsPerm(perm);
      if (perm.at(0) != 0) { CHECK_OR_RETURN(!in_tensor_desc->is_dynamic()); }
      *out_tensor_desc = *in_tensor_desc;
      FOR_RANGE(size_t, i, 0, perm.size()) { out_shape->Set(i, in_shape.At(perm[i])); }
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      if (ctx->BatchAxis4ArgNameAndIndex("input", 0)->has_value()) {
        const auto& perm = ctx->GetAttr<std::vector<int32_t>>("perm");
        ctx->BatchAxis4ArgNameAndIndex("output", 0)
            ->set_value(perm.at(ctx->BatchAxis4ArgNameAndIndex("input", 0)->value()));
      } else {
        ctx->BatchAxis4ArgNameAndIndex("output", 0)->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& input_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0);
      const auto& perm = ctx->GetAttr<std::vector<int32_t>>("perm");
      CHECK_EQ(perm.size(), input_tensor.shape().NumAxes());
      FOR_RANGE(int32_t, i, 0, perm.size()) {
        int32_t axis = perm.at(i);
        if (axis < 0) { axis += perm.size(); }
        CHECK_GE(axis, 0);
        CHECK_LT(axis, perm.size());
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), axis).Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("transpose")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("input", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        const auto& tmp = op.attr<std::vector<int32_t>>("perm");
        std::vector<int32_t> perm;
        perm.resize(tmp.size());
        FOR_RANGE(int32_t, i, 0, tmp.size()) { perm.at(tmp.at(i)) = i; }
        user_op::UserOpConfWrapper transpose_grad_op =
            builder.Op("transpose")
                .Input("input", op.GetGradTensorWithOpOutput("output", 0))
                .Output("output")
                .Attr<std::vector<int32_t>>("perm", perm)
                .Build();
        op.BindGradTensorWithOpInput(transpose_grad_op.output("output", 0), "input", 0);
        AddOp(transpose_grad_op);
      }
    });
}  // namespace oneflow