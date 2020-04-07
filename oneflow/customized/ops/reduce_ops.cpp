#include <cstdint>
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_attr.pb.h"

namespace oneflow {

REGISTER_USER_OP("reduce")
    .Input("tensor_in")
    .Output("tensor_out")
    .Attr("axis", UserOpAttrType::kAtListInt32)
    .Attr("keepdims", UserOpAttrType::kAtBool)
    .Attr("reduce_func_type", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("tensor_in", 0);
      auto axis = ctx->GetAttr<std::vector<int32_t>>("axis");
      bool keepdims = ctx->GetAttr<bool>("keepdims");
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("tensor_out", 0);
      if (axis.empty()) {
        if (keepdims) {
          *out_shape = Shape::Ones(in_shape->NumAxes());
        } else {
          *out_shape = Shape({1});
        }
      } else {
        const AxisVector axis_vec = {axis.begin(), axis.end()};
        const Shape& reduced_shape = CreateReducedShape(*in_shape, axis_vec);
        if (keepdims) {
          *out_shape = reduced_shape;
        } else {
          *out_shape = reduced_shape.RemoveOnes(axis_vec);
        }
      }
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      const auto& reduced_axes = ctx->GetAttr<std::vector<int32_t>>("axis");
      const bool keep_dims = ctx->GetAttr<bool>("keepdims");
      HashSet<int64_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
      if (ctx->BatchAxis4ArgNameAndIndex("tensor_in", 0)->has_value() && !conf_axes.empty()
          && conf_axes.find(ctx->BatchAxis4ArgNameAndIndex("tensor_in", 0)->has_value())
                 == conf_axes.end()) {
        *ctx->BatchAxis4ArgNameAndIndex("tensor_out", 0) =
            *ctx->BatchAxis4ArgNameAndIndex("tensor_in", 0);
      } else if (conf_axes.empty() && keep_dims == false) {
        ctx->BatchAxis4ArgNameAndIndex("tensor_out", 0)->set_value(0);
      } else {
        ctx->BatchAxis4ArgNameAndIndex("tensor_out", 0)->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor_in =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("tensor_in", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor_in.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });
}  // namespace oneflow