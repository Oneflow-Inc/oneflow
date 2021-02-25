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

namespace oneflow {

namespace {

int64_t ShiftNegativeAxisIfNeed(const Shape& shape, int64_t axis) {
  const int64_t shifted = axis < 0 ? axis + shape.NumAxes() : axis;
  CHECK_GE(shifted, 0);
  CHECK_LT(shifted, shape.NumAxes());
  return shifted;
}

Shape InferBnParamShape(const Shape& x_shape, const int64_t begin_norm_axis) {
  DimVector bn_param_shape_dim_vec;
  bn_param_shape_dim_vec.insert(bn_param_shape_dim_vec.end(), x_shape.dim_vec().cbegin(),
                                x_shape.dim_vec().cbegin() + begin_norm_axis);
  const Shape bn_param_shape(bn_param_shape_dim_vec);
  return bn_param_shape;
}

oneflow::DataType InferBnParamDataType(const DataType x_data_type) {
  return x_data_type == DataType::kFloat16 ? DataType::kFloat : x_data_type;
}

}  // namespace

REGISTER_USER_OP("layer_norm")
    .Input("x")
    .OptionalInput("beta")
    .OptionalInput("gamma")
    .Output("y")
    .Output("mean")
    .Output("inv_variance")
    .OptionalOutput("normalized")
    .Attr<bool>("center")
    .Attr<bool>("scale")
    .Attr<int64_t>("begin_norm_axis")
    .Attr<int64_t>("begin_params_axis")
    .Attr<double>("epsilon")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      user_op::TensorDesc* mean = ctx->TensorDesc4ArgNameAndIndex("mean", 0);
      user_op::TensorDesc* inv_variance = ctx->TensorDesc4ArgNameAndIndex("inv_variance", 0);
      const bool center = ctx->Attr<bool>("center");
      const bool scale = ctx->Attr<bool>("scale");
      const int64_t begin_params_axis =
          ShiftNegativeAxisIfNeed(x->shape(), ctx->Attr<int64_t>("begin_params_axis"));
      *y = *x;
      DimVector param_shape_dim_vec;
      param_shape_dim_vec.insert(param_shape_dim_vec.end(),
                                 x->shape().dim_vec().cbegin() + begin_params_axis,
                                 x->shape().dim_vec().cend());
      if (param_shape_dim_vec.empty()) { param_shape_dim_vec.push_back(1); }
      const Shape param_shape(param_shape_dim_vec);
      if (center) {
        const user_op::TensorDesc* beta = ctx->TensorDesc4ArgNameAndIndex("beta", 0);
        CHECK_OR_RETURN(ctx->parallel_ctx().parallel_num() == 1
                        || ctx->SbpParallel4ArgNameAndIndex("beta", 0).has_broadcast_parallel());
        CHECK_EQ_OR_RETURN(beta->shape(), param_shape);
        CHECK_EQ_OR_RETURN(beta->data_type(), x->data_type());
      }
      if (scale) {
        user_op::TensorDesc* normalized = ctx->TensorDesc4ArgNameAndIndex("normalized", 0);
        const user_op::TensorDesc* gamma = ctx->TensorDesc4ArgNameAndIndex("gamma", 0);
        CHECK_OR_RETURN(ctx->parallel_ctx().parallel_num() == 1
                        || ctx->SbpParallel4ArgNameAndIndex("gamma", 0).has_broadcast_parallel());
        CHECK_EQ_OR_RETURN(gamma->shape(), param_shape);
        CHECK_EQ_OR_RETURN(gamma->data_type(), x->data_type());
        *normalized = *x;
      }
      const int64_t begin_norm_axis =
          ShiftNegativeAxisIfNeed(x->shape(), ctx->Attr<int64_t>("begin_norm_axis"));
      *mean->mut_shape() = InferBnParamShape(x->shape(), begin_norm_axis);
      *mean->mut_data_type() = InferBnParamDataType(x->data_type());
      *inv_variance = *mean;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .Broadcast(user_op::OpArg("gamma", 0))
          .Broadcast(user_op::OpArg("beta", 0))
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("layer_norm_grad")
    .Input("dy")
    .Input("x")
    .Input("mean")
    .Input("inv_variance")
    .OptionalInput("_add_to_output")
    .Output("dx")
    .Attr<int64_t>("begin_norm_axis")
    .Attr<double>("epsilon")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* mean = ctx->TensorDesc4ArgNameAndIndex("mean", 0);
      const user_op::TensorDesc* inv_variance = ctx->TensorDesc4ArgNameAndIndex("inv_variance", 0);
      user_op::TensorDesc* dx = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
      CHECK_EQ_OR_RETURN(dy->data_type(), x->data_type());
      CHECK_EQ_OR_RETURN(dy->shape(), x->shape());
      const int64_t begin_norm_axis = ctx->Attr<int64_t>("begin_norm_axis");
      CHECK_GT(begin_norm_axis, 0);
      const DataType& bn_param_data_type = InferBnParamDataType(x->data_type());
      const Shape& bn_param_shape = InferBnParamShape(x->shape(), begin_norm_axis);
      CHECK_EQ_OR_RETURN(mean->data_type(), bn_param_data_type);
      CHECK_EQ_OR_RETURN(mean->shape(), bn_param_shape);
      CHECK_EQ_OR_RETURN(inv_variance->data_type(), bn_param_data_type);
      CHECK_EQ_OR_RETURN(inv_variance->shape(), bn_param_shape);
      *dx = *dy;
      if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
        const auto* add_to_output = ctx->TensorDesc4ArgNameAndIndex("_add_to_output", 0);
        CHECK_EQ_OR_RETURN(add_to_output->data_type(), dx->data_type());
        CHECK_EQ_OR_RETURN(add_to_output->shape(), dx->shape());
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("layer_norm_param_grad")
    .Input("dy")
    .OptionalInput("normalized")
    .OptionalInput("gamma")
    .OptionalOutput("normalized_diff")
    .OptionalOutput("beta_diff")
    .OptionalOutput("gamma_diff")
    .OptionalOutput("reduce_buf")
    .Attr<int64_t>("begin_params_axis")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // TODO: tsai: replace lambda with user op if
      auto has_tensor = [ctx](const std::string& bn) -> bool {
        bool ret = false;
        for (auto t : ctx->inputs()) {
          if (bn == t.first) { return true; }
        }
        for (auto t : ctx->outputs()) {
          if (bn == t.first) { return true; }
        }
        return ret;
      };
      const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      const int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
      const bool has_beta_diff = has_tensor("beta_diff");
      const bool has_gamma_diff = has_tensor("gamma_diff");
      const bool has_gamma = has_tensor("gamma");
      const bool has_normalized_diff = has_tensor("normalized_diff");
      if (has_beta_diff || has_gamma_diff) {
        user_op::TensorDesc* reduce_buf = ctx->TensorDesc4ArgNameAndIndex("reduce_buf", 0);
        *reduce_buf = *dy;
      }
      CHECK_GE_OR_RETURN(begin_params_axis, 1);
      CHECK_LT_OR_RETURN(begin_params_axis, dy->shape().NumAxes());
      DimVector param_shape_dim_vec;
      param_shape_dim_vec.insert(param_shape_dim_vec.end(),
                                 dy->shape().dim_vec().cbegin() + begin_params_axis,
                                 dy->shape().dim_vec().cend());
      if (param_shape_dim_vec.empty()) { param_shape_dim_vec.push_back(1); }
      const Shape param_shape(param_shape_dim_vec);
      if (has_beta_diff) {
        user_op::TensorDesc* beta_diff = ctx->TensorDesc4ArgNameAndIndex("beta_diff", 0);
        *beta_diff->mut_data_type() = dy->data_type();
        *beta_diff->mut_shape() = param_shape;
      }
      if (has_gamma_diff) {
        user_op::TensorDesc* gamma_diff = ctx->TensorDesc4ArgNameAndIndex("gamma_diff", 0);
        const user_op::TensorDesc* normalized = ctx->TensorDesc4ArgNameAndIndex("normalized", 0);
        CHECK_EQ_OR_RETURN(normalized->data_type(), normalized->data_type());
        CHECK_EQ_OR_RETURN(normalized->shape(), normalized->shape());
        *gamma_diff->mut_data_type() = dy->data_type();
        *gamma_diff->mut_shape() = param_shape;
      }
      if (has_normalized_diff) {
        user_op::TensorDesc* normalized_diff =
            ctx->TensorDesc4ArgNameAndIndex("normalized_diff", 0);
        *normalized_diff = *dy;
      }
      if (has_gamma) {
        const user_op::TensorDesc* gamma = ctx->TensorDesc4ArgNameAndIndex("gamma", 0);
        CHECK_OR_RETURN(ctx->parallel_ctx().parallel_num() == 1
                        || ctx->SbpParallel4ArgNameAndIndex("gamma", 0).has_broadcast_parallel())
            << "parallel_num: " << ctx->parallel_ctx().parallel_num() << ", "
            << "gamma sbp:" << ctx->SbpParallel4ArgNameAndIndex("gamma", 0).DebugString();
        CHECK_EQ_OR_RETURN(gamma->data_type(), dy->data_type());
        CHECK_EQ_OR_RETURN(gamma->shape(), param_shape);
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .Broadcast(user_op::OpArg("gamma", 0))
          .PartialSum(user_op::OpArg("gamma_diff", 0))
          .PartialSum(user_op::OpArg("beta_diff", 0))
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("layer_norm")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      const bool center = op.attr<bool>("center");
      const bool scale = op.attr<bool>("scale");
      const bool has_beta = center;
      const bool has_gamma = scale;
      const bool has_beta_diff = has_beta && op.NeedGenGradTensor4OpInput("beta", 0);
      const bool has_gamma_diff = has_gamma && op.NeedGenGradTensor4OpInput("gamma", 0);
      const bool need_scale_out_diff = has_gamma && op.NeedGenGradTensor4OpInput("x", 0);
      const Shape& x_shape = op.TensorDesc4ArgNameAndIndex("x", 0).shape();
      const int64_t begin_norm_axis =
          ShiftNegativeAxisIfNeed(x_shape, op.attr<int64_t>("begin_norm_axis"));
      const int64_t begin_params_axis =
          ShiftNegativeAxisIfNeed(x_shape, op.attr<int64_t>("begin_params_axis"));
      std::string dy = op.GetGradTensorWithOpOutput("y", 0);
      if (has_beta_diff || has_gamma_diff || need_scale_out_diff) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_param_grad");
        auto grad_op_builder = builder.Op("layer_norm_param_grad")
                                   .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                   .Attr("begin_params_axis", begin_params_axis);
        if (has_beta_diff) { grad_op_builder.Output("beta_diff"); }
        if (has_gamma_diff || need_scale_out_diff) {
          grad_op_builder.Input("gamma", op.input("gamma", 0));
        }
        if (has_gamma_diff) {
          grad_op_builder.Input("normalized", op.output("normalized", 0));
          grad_op_builder.Output("gamma_diff");
        }
        if (need_scale_out_diff) { grad_op_builder.Output("normalized_diff"); }
        if (has_beta_diff || has_gamma_diff) { grad_op_builder.Output("reduce_buf"); }
        auto grad_op = grad_op_builder.Build();
        if (has_beta_diff) {
          op.BindGradTensorWithOpInput(grad_op.output("beta_diff", 0), "beta", 0);
        }
        if (has_gamma_diff) {
          op.BindGradTensorWithOpInput(grad_op.output("gamma_diff", 0), "gamma", 0);
        }
        if (need_scale_out_diff) { dy = grad_op.output("normalized_diff", 0); }
        AddOp(grad_op);
      }
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("layer_norm_grad")
                .Input("x", op.input("x", 0))
                .Input("dy", dy)
                .Input("mean", op.output("mean", 0))
                .Input("inv_variance", op.output("inv_variance", 0))
                .Output("dx")
                .Attr("begin_norm_axis", begin_norm_axis)
                .Attr("epsilon", op.attr<double>("epsilon"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
