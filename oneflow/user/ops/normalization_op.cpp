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
#ifdef WITH_CUDA
#include "oneflow/core/device/cudnn_util.h"
#endif

namespace oneflow {

namespace {

std::function<Maybe<void>(const std::string&)> MakeCheckParamTensorDescFn(
    user_op::InferContext* ctx, DataType data_type, const Shape& shape) {
  return [=](const std::string& bn) -> Maybe<void> {
    if (ctx->user_op_conf().has_input(bn, 0)) {
      const auto* tensor_desc = ctx->TensorDesc4ArgNameAndIndex(bn, 0);
      CHECK_OR_RETURN(tensor_desc != nullptr);
      CHECK_EQ_OR_RETURN(tensor_desc->data_type(), data_type);
      CHECK_EQ_OR_RETURN(tensor_desc->shape(), shape);
    }
    return Maybe<void>::Ok();
  };
}

std::function<Maybe<void>(const std::string&)> MakeSetParamTensorDescFn(user_op::InferContext* ctx,
                                                                        DataType data_type,
                                                                        const Shape& shape) {
  return [=](const std::string& bn) -> Maybe<void> {
    if (ctx->user_op_conf().has_output(bn, 0)) {
      auto* tensor_desc = ctx->TensorDesc4ArgNameAndIndex(bn, 0);
      CHECK_OR_RETURN(tensor_desc != nullptr);
      *tensor_desc->mut_data_type() = data_type;
      *tensor_desc->mut_shape() = shape;
    }
    return Maybe<void>::Ok();
  };
}

void FwInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                        const user_op::UserOpConfWrapper& conf) {
  bool training;
  if (conf.op_type_name() == "normalization") {
    training = conf.attr<bool>("training");
  } else {
    training = true;
  }
  user_op::InputArgModifier* moving_mean_modifier = GetInputArgModifierFn("moving_mean", 0);
  CHECK(moving_mean_modifier != nullptr);
  moving_mean_modifier->set_is_mutable(training);
  moving_mean_modifier->set_requires_grad(false);
  user_op::InputArgModifier* moving_variance_modifier = GetInputArgModifierFn("moving_variance", 0);
  CHECK(moving_variance_modifier != nullptr);
  moving_variance_modifier->set_is_mutable(training);
  moving_variance_modifier->set_requires_grad(false);
}

Maybe<void> FwGetSbpFn(user_op::SbpContext* ctx) {
  std::vector<user_op::OpArg> split_args;
  split_args.emplace_back("x", 0);
  split_args.emplace_back("y", 0);
  if (ctx->user_op_conf().has_input("addend", 0)) { split_args.emplace_back("addend", 0); }
  if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
    split_args.emplace_back("_add_to_output", 0);
  }
  std::vector<user_op::OpArg> broadcast_args;
  broadcast_args.emplace_back("moving_mean", 0);
  broadcast_args.emplace_back("moving_variance", 0);
  broadcast_args.emplace_back("gamma", 0);
  broadcast_args.emplace_back("beta", 0);
  if (ctx->user_op_conf().has_output("mean", 0)) { broadcast_args.emplace_back("mean", 0); }
  if (ctx->user_op_conf().has_output("inv_variance", 0)) {
    broadcast_args.emplace_back("inv_variance", 0);
  }
  if (ctx->user_op_conf().has_output("reserve_space", 0)) {
    broadcast_args.emplace_back("reserve_space", 0);
  }
  ctx->NewBuilder().Broadcast(broadcast_args).Split(split_args, 0).Build();
  return Maybe<void>::Ok();
}

user_op::TensorDescInferFn MakeFwTensorDescInferFn(
    const std::function<Maybe<void>(user_op::InferContext* ctx, const user_op::TensorDesc* x,
                                    user_op::TensorDesc* reserve_space)>& reserve_space_infer_fn) {
  return [reserve_space_infer_fn](user_op::InferContext* ctx) -> Maybe<void> {
#ifdef WITH_CUDA
    // assume cudnn is enabled
    CHECK_GE_OR_RETURN(ctx->Attr<float>("epsilon"), CUDNN_BN_MIN_EPSILON);
#endif
    const auto* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
    const auto data_type = x->data_type();
    const Shape& x_shape = x->shape();
    if (ctx->user_op_conf().has_input("addend", 0)) {
      const auto* addend = ctx->TensorDesc4ArgNameAndIndex("addend", 0);
      CHECK_EQ_OR_RETURN(addend->data_type(), data_type);
      CHECK_EQ_OR_RETURN(addend->shape(), x_shape);
    }
    if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
      const auto* add_to_output = ctx->TensorDesc4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ_OR_RETURN(add_to_output->data_type(), data_type);
      CHECK_EQ_OR_RETURN(add_to_output->shape(), x_shape);
    }
    *ctx->TensorDesc4ArgNameAndIndex("y", 0) = *x;
    const auto axis = ctx->Attr<int32_t>("axis");
    CHECK_GE_OR_RETURN(axis, 0);
    CHECK_LT_OR_RETURN(axis, x_shape.NumAxes());
    const Shape param_shape({x_shape.At(axis)});
    const DataType param_data_type = data_type == DataType::kFloat16 ? DataType::kFloat : data_type;
    const auto CheckParamTensorDesc = MakeCheckParamTensorDescFn(ctx, param_data_type, param_shape);
    const auto SetParamTensorDesc = MakeSetParamTensorDescFn(ctx, param_data_type, param_shape);
    JUST(CheckParamTensorDesc("moving_mean"));
    JUST(CheckParamTensorDesc("moving_variance"));
    JUST(CheckParamTensorDesc("beta"));
    JUST(CheckParamTensorDesc("gamma"));
    JUST(SetParamTensorDesc("mean"));
    JUST(SetParamTensorDesc("inv_variance"));
    if (ctx->user_op_conf().has_output("reserve_space", 0)) {
      CHECK(reserve_space_infer_fn);
      reserve_space_infer_fn(ctx, x, ctx->TensorDesc4ArgNameAndIndex("reserve_space", 0));
    }
    return Maybe<void>::Ok();
  };
}

user_op::TensorDescInferFn MakeFwTensorDescInferFn() {
  return MakeFwTensorDescInferFn(
      std::function<Maybe<void>(user_op::InferContext * ctx, const user_op::TensorDesc* x,
                                user_op::TensorDesc* reserve_space)>());
}

REGISTER_USER_OP("normalization")
    .Input("x")
    .Input("moving_mean")
    .Input("moving_variance")
    .Input("gamma")
    .Input("beta")
    .OptionalInput("_add_to_output")
    .Output("y")
    .OptionalOutput("mean")
    .OptionalOutput("inv_variance")
    .Attr<int32_t>("axis")
    .Attr<float>("epsilon")
    .Attr<bool>("training")
    .Attr<float>("momentum")
    .SetInputArgModifyFn(FwInputArgModifyFn)
    .SetTensorDescInferFn(MakeFwTensorDescInferFn())
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP("normalization_add_relu")
    .Input("x")
    .OptionalInput("addend")
    .Input("moving_mean")
    .Input("moving_variance")
    .Input("gamma")
    .Input("beta")
    .Output("y")
    .Output("reserve_space")
    .OptionalOutput("mean")
    .OptionalOutput("inv_variance")
    .Attr<int32_t>("axis")
    .Attr<float>("epsilon")
    .Attr<float>("momentum")
    .SetInputArgModifyFn(FwInputArgModifyFn)
    .SetTensorDescInferFn(
        MakeFwTensorDescInferFn([](user_op::InferContext* ctx, const user_op::TensorDesc* x,
                                   user_op::TensorDesc* reserve_space) -> Maybe<void> {
          const auto* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
          *reserve_space->mut_data_type() = DataType::kInt32;
          *reserve_space->mut_shape() =
              Shape({static_cast<int64_t>(RoundUp(x_desc->shape().elem_cnt(), 32) / 32)});
          return Maybe<void>::Ok();
        }))
    .SetGetSbpFn(FwGetSbpFn);

#if defined(WITH_CUDA) && (CUDNN_VERSION >= 7401)

REGISTER_USER_OP("cudnn_fused_normalization_add_relu")
    .Input("x")
    .OptionalInput("addend")
    .Input("moving_mean")
    .Input("moving_variance")
    .Input("gamma")
    .Input("beta")
    .Output("y")
    .Output("reserve_space")
    .OptionalOutput("mean")
    .OptionalOutput("inv_variance")
    .Attr<int32_t>("axis")
    .Attr<float>("epsilon")
    .Attr<float>("momentum")
    .SetInputArgModifyFn(FwInputArgModifyFn)
    .SetTensorDescInferFn(
        MakeFwTensorDescInferFn([](user_op::InferContext* ctx, const user_op::TensorDesc* x,
                                   user_op::TensorDesc* reserve_space) -> Maybe<void> {
          const Shape& x_shape = x->shape();
          const auto axis = ctx->Attr<int32_t>("axis");
          CHECK_EQ_OR_RETURN(x_shape.Count(axis + 1), 1);
          int64_t n = x_shape.At(0);
          int64_t h = x_shape.Count(1, axis);
          int64_t w = 1;
          int64_t c = x_shape.At(axis);
          cudnnHandle_t cudnn_handle;
          size_t reserve_space_size;
          OF_CUDNN_CHECK(cudnnCreate(&cudnn_handle));
          CudnnTensorDesc xy_desc(CUDNN_TENSOR_NHWC, x->data_type(), n, c, h, w);
          CudnnActivationDesc activation_desc(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);
          cudnnBatchNormOps_t ops;
          if (ctx->user_op_conf().has_input("addend", 0)) {
            ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
          } else {
            ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
          }
          OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
              cudnn_handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops, activation_desc.Get(),
              xy_desc.Get(), &reserve_space_size));
          reserve_space_size = std::max(reserve_space_size, GetOneVal<size_t>());
          *reserve_space->mut_data_type() = DataType::kChar;
          *reserve_space->mut_shape() = Shape({static_cast<int64_t>(reserve_space_size)});
          OF_CUDNN_CHECK(cudnnDestroy(cudnn_handle));
          return Maybe<void>::Ok();
        }))
    .SetGetSbpFn(FwGetSbpFn);

#endif

Maybe<void> BwTensorDescInferFn(user_op::InferContext* ctx) {
#ifdef WITH_CUDA
  // assume cudnn is enabled
  CHECK_GE_OR_RETURN(ctx->Attr<float>("epsilon"), CUDNN_BN_MIN_EPSILON);
#endif
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const DataType x_type = x->data_type();
  const Shape& x_shape = x->shape();
  const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
  CHECK_EQ_OR_RETURN(dy->data_type(), x_type);
  CHECK_EQ_OR_RETURN(dy->shape(), x_shape);
  if (ctx->user_op_conf().has_input("y", 0)) {
    const user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
    CHECK_EQ_OR_RETURN(y->data_type(), x_type);
    CHECK_EQ_OR_RETURN(y->shape(), x_shape);
  }
  *ctx->TensorDesc4ArgNameAndIndex("dx", 0) = *x;
  if (ctx->user_op_conf().has_output("addend_diff", 0)) {
    *ctx->TensorDesc4ArgNameAndIndex("addend_diff", 0) = *x;
  }
  const Shape param_shape({x_shape.At(ctx->Attr<int32_t>("axis"))});
  const DataType param_data_type = x_type == DataType::kFloat16 ? DataType::kFloat : x_type;
  const auto CheckParamTensorDesc = MakeCheckParamTensorDescFn(ctx, param_data_type, param_shape);
  const auto SetParamTensorDesc = MakeSetParamTensorDescFn(ctx, param_data_type, param_shape);
  JUST(CheckParamTensorDesc("mean"));
  JUST(CheckParamTensorDesc("inv_variance"));
  JUST(CheckParamTensorDesc("gamma"));
  JUST(CheckParamTensorDesc("beta"));
  JUST(SetParamTensorDesc("gamma_diff"));
  JUST(SetParamTensorDesc("beta_diff"));
  return Maybe<void>::Ok();
}

Maybe<void> BwGetSbpFn(user_op::SbpContext* ctx) {
  std::vector<user_op::OpArg> broadcast_args;
  broadcast_args.emplace_back("mean", 0);
  broadcast_args.emplace_back("inv_variance", 0);
  broadcast_args.emplace_back("gamma", 0);
  if (ctx->user_op_conf().has_input("beta", 0)) { broadcast_args.emplace_back("beta", 0); }
  if (ctx->user_op_conf().has_input("reserve_space", 0)) {
    broadcast_args.emplace_back("reserve_space", 0);
  }
  std::vector<user_op::OpArg> partial_sum_args;
  partial_sum_args.emplace_back("gamma_diff", 0);
  partial_sum_args.emplace_back("beta_diff", 0);
  std::vector<user_op::OpArg> split_args;
  split_args.emplace_back("x", 0);
  split_args.emplace_back("dy", 0);
  split_args.emplace_back("dx", 0);
  if (ctx->user_op_conf().has_input("y", 0)) { split_args.emplace_back("y", 0); }
  if (ctx->user_op_conf().has_output("addend_diff", 0)) {
    split_args.emplace_back("addend_diff", 0);
  }
  ctx->NewBuilder()
      .Broadcast(broadcast_args)
      .PartialSum(partial_sum_args)
      .Split(split_args, 0)
      .Build();
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("normalization_grad")
    .Input("x")
    .Input("dy")
    .Input("mean")
    .Input("inv_variance")
    .Input("gamma")
    .Output("gamma_diff")
    .Output("beta_diff")
    .Output("dx")
    .Attr<int32_t>("axis")
    .Attr<float>("epsilon")
    .SetTensorDescInferFn(BwTensorDescInferFn)
    .SetGetSbpFn(BwGetSbpFn);

REGISTER_USER_OP("normalization_add_relu_grad")
    .Input("x")
    .Input("dy")
    .Input("mean")
    .Input("inv_variance")
    .Input("gamma")
    .Input("beta")
    .Input("reserve_space")
    .Input("y")
    .Output("gamma_diff")
    .Output("beta_diff")
    .Output("dx")
    .OptionalOutput("addend_diff")
    .Attr<int32_t>("axis")
    .Attr<float>("epsilon")
    .SetTensorDescInferFn(BwTensorDescInferFn)
    .SetGetSbpFn(BwGetSbpFn);

#if defined(WITH_CUDA) && (CUDNN_VERSION >= 7401)

REGISTER_USER_OP("cudnn_fused_normalization_add_relu_grad")
    .Input("x")
    .Input("dy")
    .Input("mean")
    .Input("inv_variance")
    .Input("gamma")
    .Input("beta")
    .Input("reserve_space")
    .Input("y")
    .Output("gamma_diff")
    .Output("beta_diff")
    .Output("dx")
    .OptionalOutput("addend_diff")
    .Attr<int32_t>("axis")
    .Attr<float>("epsilon")
    .SetTensorDescInferFn(BwTensorDescInferFn)
    .SetGetSbpFn(BwGetSbpFn);

#endif

REGISTER_USER_OP_GRAD("normalization")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      const bool is_training = ctx->FwOp().attr<bool>("training");
      const bool is_fp16 = ctx->FwOp().arg_tensor_desc("y", 0).data_type() == DataType::kFloat16;

      // calculate inv_variance from moving_variance
      const auto var_add_eps_op_name =
          "System-AutoGrad-" + ctx->FwOp().op_name() + "-VarianceAddEpsilon";
      ctx->DefineOp(var_add_eps_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("scalar_add")
            .InputBind("in", ctx->FwOp().input("moving_variance", 0))
            .Attr("has_float_operand", true)
            .Attr("has_int_operand", false)
            .Attr("int_operand", static_cast<int64_t>(0))
            .Attr("float_operand", static_cast<double>(ctx->FwOp().attr<float>("epsilon")))
            .Output("out")
            .Build();
      });

      const auto var_rsqrt_op_name = "System-AutoGrad-" + ctx->FwOp().op_name() + "-VarianceRsqrt";
      ctx->DefineOp(var_rsqrt_op_name,
                    [&ctx, &var_add_eps_op_name](user_op::BackwardOpBuilder& builder) {
                      return builder.OpTypeName("rsqrt")
                          .InputBind("x", ctx->GetOp(var_add_eps_op_name).output("out", 0))
                          .Output("y")
                          .Build();
                    });

      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      ctx->DefineOp(grad_op_name, [&ctx, &is_training,
                                   &var_rsqrt_op_name](user_op::BackwardOpBuilder& builder) {
        builder.OpTypeName("normalization_grad")
            .InputBind("x", ctx->FwOp().input("x", 0))
            .InputBind("dy", ctx->FwOp().output_grad("y", 0))
            .InputBind("gamma", ctx->FwOp().input("gamma", 0))
            .Attr("axis", ctx->FwOp().attr<int32_t>("axis"))
            .Attr("epsilon", ctx->FwOp().attr<float>("epsilon"))
            .Output("gamma_diff")
            .Output("beta_diff")
            .Output("dx");
        if (is_training) {
          builder.InputBind("mean", ctx->FwOp().output("mean", 0))
              .InputBind("inv_variance", ctx->FwOp().output("inv_variance", 0));
        } else {
          builder.InputBind("mean", ctx->FwOp().input("moving_mean", 0))
              .InputBind("inv_variance", ctx->GetOp(var_rsqrt_op_name).output("y", 0));
        }
        return builder.Build();
      });

      // calculate dx manually as cudnn cannot be used in evaluation mode
      // reference: https://github.com/pytorch/pytorch/issues/4284
      const auto axis = ctx->FwOp().attr<int32_t>("axis");
      const auto BroadcastMulAtAxisOpDefine =
          [&ctx, &axis](std::function<std::string()> scale_bn_func,
                        std::function<std::string()> input_bn_func, const std::string& name) {
            // local variable(scale_bn_func) need to be captured by value
            const auto reshape_op_name = "System-AutoGrad-" + name + "-Reshape";
            ctx->DefineOp(reshape_op_name,
                          [&ctx, &axis, scale_bn_func](user_op::BackwardOpBuilder& builder) {
                            DimVector broadcast_dim_vec;
                            const auto& in_shape = ctx->FwOp().arg_tensor_desc("x", 0).shape();
                            FOR_RANGE(size_t, i, 0, in_shape.NumAxes()) {
                              if (i != axis) {
                                broadcast_dim_vec.push_back(1);
                              } else {
                                broadcast_dim_vec.push_back(in_shape.At(axis));
                              }
                            }
                            const Shape broadcast_shape(broadcast_dim_vec);

                            return builder.OpTypeName("reshape")
                                .InputBind("in", scale_bn_func())
                                .Attr("shape", broadcast_shape)
                                .Output("out")
                                .Build();
                          });

            // local variable(reshape_op_name/input_bn_func) need to be captured by value
            const auto mul_op_name = "System-AutoGrad-" + name + "-BroadcastMul";
            ctx->DefineOp(mul_op_name, [&ctx, reshape_op_name,
                                        input_bn_func](user_op::BackwardOpBuilder& builder) {
              return builder.OpTypeName("broadcast_mul")
                  .InputBind("x", ctx->GetOp(reshape_op_name).output("out", 0))
                  .InputBind("y", input_bn_func())
                  .Output("z")
                  .Build();
            });
          };

      const auto dy_h2f_cast_op_name = "System-AutoGrad-" + ctx->FwOp().op_name() + "-Cast-dy-h2f";
      ctx->DefineOp(dy_h2f_cast_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("cast")
            .Input("in", ctx->FwOp().output_grad("y", 0))
            .Output("out")
            .Attr("dtype", ctx->FwOp().arg_tensor_desc("gamma", 0).data_type())
            .Build();
      });

      const std::string mul_gamma_name = ctx->FwOp().op_name() + "_out_grad_mul_gamma";
      const auto dy_mul_gamma_op_name = "System-AutoGrad-" + mul_gamma_name + "-BroadcastMul";
      BroadcastMulAtAxisOpDefine([&ctx]() { return ctx->FwOp().input("gamma", 0); },
                                 [&ctx, &is_fp16, &dy_h2f_cast_op_name]() {
                                   if (is_fp16) {
                                     return ctx->GetOp(dy_h2f_cast_op_name).output("out", 0);
                                   } else {
                                     return ctx->FwOp().output_grad("y", 0);
                                   }
                                 },
                                 mul_gamma_name);

      const std::string mul_inv_var_name = ctx->FwOp().op_name() + "_out_grad_mul_inv_var";
      const auto dy_mul_inv_var_op_name = "System-AutoGrad-" + mul_inv_var_name + "-BroadcastMul";
      BroadcastMulAtAxisOpDefine(
          [&ctx, &var_rsqrt_op_name]() { return ctx->GetOp(var_rsqrt_op_name).output("y", 0); },
          [&ctx, &dy_mul_gamma_op_name]() {
            return ctx->GetOp(dy_mul_gamma_op_name).output("z", 0);
          },
          mul_inv_var_name);

      const auto dx_f2h_cast_op_name = "System-AutoGrad-" + ctx->FwOp().op_name() + "-Cast-dx-f2h";
      ctx->DefineOp(dx_f2h_cast_op_name,
                    [&ctx, &dy_mul_inv_var_op_name](user_op::BackwardOpBuilder& builder) {
                      return builder.OpTypeName("cast")
                          .InputBind("in", ctx->GetOp(dy_mul_inv_var_op_name).output("z", 0))
                          .Output("out")
                          .Attr("dtype", DataType::kFloat16)
                          .Build();
                    });

      // TODO(liujuncheng): delete identity op when boxing support separated regsts
      const auto gamma_identity_op_name = ctx->FwOp().op_name() + "_grad_gamma_diff_identity";
      ctx->DefineOp(gamma_identity_op_name,
                    [&ctx, &grad_op_name](user_op::BackwardOpBuilder& builder) {
                      return builder.OpTypeName("identity")
                          .InputBind("in", ctx->GetOp(grad_op_name).output("gamma_diff", 0))
                          .Output("out")
                          .Build();
                    });

      // TODO(liujuncheng): delete identity op when boxing support separated regsts
      const auto beta_identity_op_name = ctx->FwOp().op_name() + "_grad_beta_diff_identity";
      ctx->DefineOp(beta_identity_op_name,
                    [&ctx, &grad_op_name](user_op::BackwardOpBuilder& builder) {
                      return builder.OpTypeName("identity")
                          .InputBind("in", ctx->GetOp(grad_op_name).output("beta_diff", 0))
                          .Output("out")
                          .Build();
                    });

      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0),
                                [&ctx, &is_training, &is_fp16, &grad_op_name, &dx_f2h_cast_op_name,
                                 &dy_mul_inv_var_op_name]() -> const std::string& {
                                  if (is_training) {
                                    return ctx->GetOp(grad_op_name).output("dx", 0);
                                  } else {
                                    if (is_fp16) {
                                      return ctx->GetOp(dx_f2h_cast_op_name).output("out", 0);
                                    } else {
                                      return ctx->GetOp(dy_mul_inv_var_op_name).output("z", 0);
                                    }
                                  }
                                });

      ctx->FwOp().InputGradBind(user_op::OpArg("gamma", 0),
                                [&ctx, &gamma_identity_op_name]() -> const std::string& {
                                  return ctx->GetOp(gamma_identity_op_name).output("out", 0);
                                });
      ctx->FwOp().InputGradBind(user_op::OpArg("beta", 0),
                                [&ctx, &beta_identity_op_name]() -> const std::string& {
                                  return ctx->GetOp(beta_identity_op_name).output("out", 0);
                                });
    });

REGISTER_USER_OP_GRAD("normalization_add_relu")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      ctx->DefineOp(grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        builder.OpTypeName("normalization_add_relu_grad")
            .InputBind("x", ctx->FwOp().input("x", 0))
            .InputBind("dy", ctx->FwOp().output_grad("y", 0))
            .InputBind("gamma", ctx->FwOp().input("gamma", 0))
            .InputBind("beta", ctx->FwOp().input("beta", 0))
            .InputBind("reserve_space", ctx->FwOp().output("reserve_space", 0))
            .InputBind("mean", ctx->FwOp().output("mean", 0))
            .InputBind("inv_variance", ctx->FwOp().output("inv_variance", 0))
            .InputBind("y", ctx->FwOp().output("y", 0))
            .Attr("axis", ctx->FwOp().attr<int32_t>("axis"))
            .Attr("epsilon", ctx->FwOp().attr<float>("epsilon"))
            .Output("gamma_diff")
            .Output("beta_diff")
            .Output("dx");
        if (ctx->FwOp().input_size("addend") != 0) { builder.Output("addend_diff"); }
        return builder.Build();
      });

      // TODO(liujuncheng): delete identity op when boxing support separated regsts
      const auto gamma_identity_op_name = ctx->FwOp().op_name() + "_grad_gamma_diff_identity";
      ctx->DefineOp(gamma_identity_op_name,
                    [&ctx, &grad_op_name](user_op::BackwardOpBuilder& builder) {
                      return builder.OpTypeName("identity")
                          .InputBind("in", ctx->GetOp(grad_op_name).output("gamma_diff", 0))
                          .Output("out")
                          .Build();
                    });

      // TODO(liujuncheng): delete identity op when boxing support separated regsts
      const auto beta_identity_op_name = ctx->FwOp().op_name() + "_grad_beta_diff_identity";
      ctx->DefineOp(beta_identity_op_name,
                    [&ctx, &grad_op_name](user_op::BackwardOpBuilder& builder) {
                      return builder.OpTypeName("identity")
                          .InputBind("in", ctx->GetOp(grad_op_name).output("beta_diff", 0))
                          .Output("out")
                          .Build();
                    });

      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0),
                                [&ctx, &grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(grad_op_name).output("dx", 0);
                                });
      if (ctx->FwOp().user_op_conf().has_input("addend", 0)) {
        ctx->FwOp().InputGradBind(user_op::OpArg("addend", 0),
                                  [&ctx, &grad_op_name]() -> const std::string& {
                                    return ctx->GetOp(grad_op_name).output("addend_diff", 0);
                                  });
      }
      ctx->FwOp().InputGradBind(user_op::OpArg("gamma", 0),
                                [&ctx, &gamma_identity_op_name]() -> const std::string& {
                                  return ctx->GetOp(gamma_identity_op_name).output("out", 0);
                                });
      ctx->FwOp().InputGradBind(user_op::OpArg("beta", 0),
                                [&ctx, &beta_identity_op_name]() -> const std::string& {
                                  return ctx->GetOp(beta_identity_op_name).output("out", 0);
                                });
    });

}  // namespace

}  // namespace oneflow
