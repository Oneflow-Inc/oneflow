/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/gather_dim_kernel_util.h"

namespace oneflow {

namespace user_op {
REGISTER_USER_OP("gather_dim")
    .Input("input")
    .Input("index")
    .Output("output")
    .Attr("dim", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("input", 0);
      int64_t input_num_axes = in->shape().NumAxes();
      CHECK_GT_OR_RETURN(input_num_axes, 0);
      CHECK_LE_OR_RETURN(input_num_axes, MAX_DIM_COUNT);

      const TensorDesc* index = ctx->TensorDesc4ArgNameAndIndex("index", 0);
      int64_t index_num_axes = index->shape().NumAxes();
      CHECK_GT_OR_RETURN(index_num_axes, 0);
      CHECK_LE_OR_RETURN(index_num_axes, MAX_DIM_COUNT);
      CHECK_OR_RETURN(IsIndexDataType(index->data_type()));

      const int64_t dim = ctx->Attr<int64_t>("dim");
      CHECK_GE_OR_RETURN(dim, 0);
      CHECK_EQ_OR_RETURN(input_num_axes, index_num_axes);

      // split_axs should NOT equals dim when in consistent view
      const SbpParallel& in_sbp = ctx->SbpParallel4ArgNameAndIndex("input", 0);
      int64_t split_axis = in_sbp.split_parallel().axis();
      auto parr_num = ctx->parallel_ctx().parallel_num();
      auto is_split = in_sbp.has_split_parallel();
      if (parr_num != 1 && is_split){
        CHECK_NE_OR_RETURN(split_axis, dim) << "split_axis should NOT equal dim";
      }

      CHECK_OR_RETURN(!in->is_dynamic());
      CHECK_OR_RETURN(!index->is_dynamic());
      
      FOR_RANGE(int64_t, i, 0, input_num_axes) {
        if (i == dim) { continue; }
        CHECK_EQ_OR_RETURN(in->shape().At(i), index->shape().At(i));
      }

      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("output", 0);
      *out->mut_shape() = index->shape();
      *out->mut_data_type() = in->data_type();

      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("index", 0);
      CHECK(indices_modifier != nullptr);
      indices_modifier->set_requires_grad(false);
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      OptInt64* indices_batch_axis = ctx->BatchAxis4ArgNameAndIndex("index", 0);
      if (indices_batch_axis->has_value()) {
        CHECK_GE_OR_RETURN(indices_batch_axis->value(), 0);
        CHECK_LE_OR_RETURN(
            indices_batch_axis->value(),
            ctx->LogicalTensorDesc4InputArgNameAndIndex("index", 0).shape().NumAxes() - 1);
      }
      *ctx->BatchAxis4ArgNameAndIndex("output", 0) = *indices_batch_axis;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& index_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("index", 0);
      int64_t index_num_axes = index_tensor.shape().NumAxes();
      const int64_t dim = ctx->Attr<int64_t>("dim");

      FOR_RANGE(int64_t, i, 0, index_num_axes - 1) {
        if (i != dim) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("index", 0), i)
              .Split(user_op::OpArg("input", 0), i)
              .Split(user_op::OpArg("output", 0), i)
              .Build();
        } else if (i == dim) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("input", 0), i)
              .Broadcast(user_op::OpArg("index", 0))
              .Broadcast(user_op::OpArg("output", 0))
              .Build();
          ctx->NewBuilder()
              .Broadcast(user_op::OpArg("input", 0))
              .Split(user_op::OpArg("index", 0), i)
              .Split(user_op::OpArg("output", 0), i)
              .Build();
        }
      }

      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("input", 0))
          .Broadcast(user_op::OpArg("index", 0))
          .Broadcast(user_op::OpArg("output", 0))
          .Build();

      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("input", 0))
          .PartialSum(user_op::OpArg("index", 0))
          .PartialSum(user_op::OpArg("output", 0))
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("scatter_dim_add_like")
    .Input("like")
    .Input("src")
    .Input("index")
    .Output("output")
    .Attr("dim", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("src", 0);
      const TensorDesc* index = ctx->TensorDesc4ArgNameAndIndex("index", 0);
      const TensorDesc* like = ctx->TensorDesc4ArgNameAndIndex("like", 0);

      CHECK_OR_RETURN(src != nullptr);
      CHECK_OR_RETURN(index != nullptr);
      CHECK_OR_RETURN(like != nullptr);

      const Shape& params_shape = like->shape();
      int64_t dim = ctx->Attr<int64_t>("dim");

      const SbpParallel& src_sbp = ctx->SbpParallel4ArgNameAndIndex("src", 0);
      int64_t split_axis = src_sbp.split_parallel().axis();
      if (ctx->parallel_ctx().parallel_num() != 1 && src_sbp.has_split_parallel()){
        CHECK_NE_OR_RETURN(split_axis, dim) << "split_axis should NOT equal dim";
      }

      int64_t src_num_axes = src->shape().NumAxes();
      CHECK_GT_OR_RETURN(src_num_axes, 0);
      CHECK_LE_OR_RETURN(src_num_axes, MAX_DIM_COUNT);

      int64_t index_num_axes = index->shape().NumAxes();
      CHECK_EQ_OR_RETURN(src_num_axes, index_num_axes);
      CHECK_LE_OR_RETURN(src_num_axes, params_shape.NumAxes());

      FOR_RANGE(int64_t, i, 0, src_num_axes) {
        CHECK_LE_OR_RETURN(index->shape().At(i), src->shape().At(i));
      }

      FOR_RANGE(int64_t, i, 0, src_num_axes) {
        if (i == dim) { continue; }
        CHECK_LE_OR_RETURN(index->shape().At(i), params_shape.At(i));
      }

      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("output", 0);
      *out->mut_shape() = params_shape;
      *out->mut_data_type() = src->data_type();

      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
      CHECK(like_arg_modifier != nullptr);
      like_arg_modifier->set_use_header_only(true);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& index_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("index", 0);
      int64_t index_num_axes = index_tensor.shape().NumAxes();
      const int64_t dim = ctx->Attr<int64_t>("dim");

      FOR_RANGE(int64_t, i, 0, index_num_axes - 1) {
        if (i != dim) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("index", 0), i)
              .Split(user_op::OpArg("src", 0), i)
              .Split(user_op::OpArg("output", 0), i)
              .Split(user_op::OpArg("like", 0), i)
              .Build();
        }
      }

      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("src", 0))
          .Broadcast(user_op::OpArg("index", 0))
          .PartialSum(user_op::OpArg("output", 0))
          .PartialSum(user_op::OpArg("like", 0))
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("gather_dim")
  .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {

    const auto op_grad_name = ctx->FwOp().op_name() + "_grad";

    ctx->DefineOp(op_grad_name, [&ctx](user_op::BackwardOpBuilder& builder) {
      return builder
          .OpTypeName(
              "scatter_dim_add_like")  // scatter_dim_add_like(like, dim, index, src) -> output
          .InputBind("index", ctx->FwOp().input("index", 0))  // scatter.index <- gather.index
          .InputBind("src",
                    ctx->FwOp().output_grad("output", 0))  // scatter.src <- grad of gather.out
          .InputBind("like", ctx->FwOp().input("input", 0))
          .Output("output")
          .Attr("dim", ctx->FwOp().attr<int64_t>("dim"))
          .Build();
    });

    ctx->FwOp().InputGradBind(user_op::OpArg("input", 0),
                              [&ctx, &op_grad_name]() -> const std::string& {
                                return ctx->GetOp(op_grad_name).output("output", 0);
                              });
});

}  // namespace user_op

}  // namespace oneflow
