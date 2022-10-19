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
#include "oneflow/core/framework/op_generated.h"
#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/cudnn_util.h"
#endif

namespace oneflow {

namespace {

std::function<Maybe<void>(const std::string&)> MakeCheckParamTensorDescFn(
    user_op::InferContext* ctx, const Shape& shape) {
  return [=](const std::string& bn) -> Maybe<void> {
    if (ctx->has_input(bn, 0)) {
      const auto& tensor_desc = ctx->InputTensorDesc(bn, 0);
      CHECK_EQ_OR_RETURN(tensor_desc.shape(), shape);
    }
    return Maybe<void>::Ok();
  };
}

std::function<Maybe<void>(const std::string&)> MakeCheckParamDataTypeFn(user_op::InferContext* ctx,
                                                                        DataType data_type) {
  return [=](const std::string& bn) -> Maybe<void> {
    if (ctx->has_input(bn, 0)) {
      const auto& tensor_desc = ctx->InputTensorDesc(bn, 0);
      CHECK_EQ_OR_RETURN(tensor_desc.data_type(), data_type)
          << "InferDataType Failed. Expected " << DataType_Name(tensor_desc.data_type())
          << ", but got " << DataType_Name(data_type);
    }
    return Maybe<void>::Ok();
  };
}

std::function<Maybe<void>(const std::string&)> MakeSetParamTensorDescFn(user_op::InferContext* ctx,
                                                                        const Shape& shape) {
  return [=](const std::string& bn) -> Maybe<void> {
    if (ctx->has_output(bn, 0)) {
      auto* tensor_desc = ctx->MutOutputTensorDesc(bn, 0);
      CHECK_OR_RETURN(tensor_desc != nullptr);
      tensor_desc->set_shape(shape);
    }
    return Maybe<void>::Ok();
  };
}

std::function<Maybe<void>(const std::string&)> MakeSetParamDataTypeFn(user_op::InferContext* ctx,
                                                                      DataType data_type) {
  return [=](const std::string& bn) -> Maybe<void> {
    if (ctx->has_output(bn, 0)) {
      auto* tensor_desc = ctx->MutOutputTensorDesc(bn, 0);
      CHECK_OR_RETURN(tensor_desc != nullptr);
      tensor_desc->set_data_type(data_type);
    }
    return Maybe<void>::Ok();
  };
}

Maybe<void> FwInputArgModifyFn(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                               const user_op::UserOpConfWrapper& conf) {
  bool training;
  if (conf.op_type_name() == "normalization") {
    training = conf.attr<bool>("training");
  } else {
    training = true;
  }
  if (conf.has_input("moving_mean", 0)) {
    CHECK_OR_RETURN(conf.has_input("moving_variance", 0));
    user_op::InputArgModifier* moving_mean_modifier = GetInputArgModifierFn("moving_mean", 0);
    CHECK_OR_RETURN(moving_mean_modifier != nullptr);
    moving_mean_modifier->set_is_mutable(training);
    moving_mean_modifier->set_requires_grad(false);
    user_op::InputArgModifier* moving_variance_modifier =
        GetInputArgModifierFn("moving_variance", 0);
    CHECK_OR_RETURN(moving_variance_modifier != nullptr);
    moving_variance_modifier->set_is_mutable(training);
    moving_variance_modifier->set_requires_grad(false);
  } else {
    CHECK_OR_RETURN(training)
        << "Must have moving mean and moving variance for normalization in inference mode.";
  }
  return Maybe<void>::Ok();
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
    const auto& x = ctx->InputTensorDesc("x", 0);
    const auto data_type = x.data_type();
    const Shape& x_shape = x.shape();
    if (ctx->has_input("addend", 0)) {
      const auto& addend = ctx->InputTensorDesc("addend", 0);
      CHECK_EQ_OR_RETURN(addend.data_type(), data_type)
          << "InferDataType Failed. Expected " << DataType_Name(addend.data_type()) << ", but got "
          << DataType_Name(data_type);
      CHECK_EQ_OR_RETURN(addend.shape(), x_shape);
    }
    if (ctx->has_input("_add_to_output", 0)) {
      const auto& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
      CHECK_EQ_OR_RETURN(add_to_output.data_type(), data_type)
          << "InferDataType Failed. Expected " << DataType_Name(add_to_output.data_type())
          << ", but got " << DataType_Name(data_type);
      CHECK_EQ_OR_RETURN(add_to_output.shape(), x_shape);
    }
    *ctx->MutOutputTensorDesc("y", 0) = x;
    const auto axis = ctx->Attr<int32_t>("axis");
    CHECK_GE_OR_RETURN(axis, 0);
    CHECK_LT_OR_RETURN(axis, x_shape.NumAxes());
    const Shape param_shape({x_shape.At(axis)});
    const auto CheckParamTensorDesc = MakeCheckParamTensorDescFn(ctx, param_shape);
    const auto SetParamTensorDesc = MakeSetParamTensorDescFn(ctx, param_shape);
    if (ctx->has_input("moving_mean", 0)) {
      CHECK_OR_RETURN(ctx->has_input("moving_variance", 0));
      JUST(CheckParamTensorDesc("moving_mean"));
      JUST(CheckParamTensorDesc("moving_variance"));
    }
    JUST(CheckParamTensorDesc("beta"));
    JUST(CheckParamTensorDesc("gamma"));
    JUST(SetParamTensorDesc("mean"));
    JUST(SetParamTensorDesc("inv_variance"));
    if (ctx->has_output("reserve_space", 0)) {
      CHECK_OR_RETURN(reserve_space_infer_fn);
      reserve_space_infer_fn(ctx, &x, ctx->MutOutputTensorDesc("reserve_space", 0));
    }
    return Maybe<void>::Ok();
  };
}

user_op::DataTypeInferFn MakeFwDataTypeInferFn(
    const std::function<Maybe<void>(user_op::InferContext* ctx, const user_op::TensorDesc* x,
                                    user_op::TensorDesc* reserve_space)>& reserve_space_infer_fn) {
  return [reserve_space_infer_fn](user_op::InferContext* ctx) -> Maybe<void> {
    const auto& x = ctx->InputTensorDesc("x", 0);
    const auto data_type = x.data_type();
    if (ctx->has_input("addend", 0)) {
      const auto& addend = ctx->InputTensorDesc("addend", 0);
      CHECK_EQ_OR_RETURN(addend.data_type(), data_type)
          << "InferDataType Failed. Expected " << DataType_Name(data_type) << ", but got "
          << DataType_Name(addend.data_type());
    }
    if (ctx->has_input("_add_to_output", 0)) {
      const auto& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
      CHECK_EQ_OR_RETURN(add_to_output.data_type(), data_type)
          << "InferDataType Failed. Expected " << DataType_Name(data_type) << ", but got "
          << DataType_Name(add_to_output.data_type());
    }
    *ctx->MutOutputTensorDesc("y", 0) = x;
    const DataType param_data_type =
        (data_type == DataType::kFloat16 || data_type == DataType::kBFloat16) ? DataType::kFloat
                                                                              : data_type;
    const auto CheckParamDataType = MakeCheckParamDataTypeFn(ctx, param_data_type);
    const auto SetParamDataType = MakeSetParamDataTypeFn(ctx, param_data_type);
    if (ctx->has_input("moving_mean", 0)) {
      CHECK_OR_RETURN(ctx->has_input("moving_variance", 0));
      JUST(CheckParamDataType("moving_mean"));
      JUST(CheckParamDataType("moving_variance"));
    }
    CHECK_OR_RETURN(ctx->has_input("gamma", 0));
    JUST(CheckParamDataType("beta"));
    JUST(CheckParamDataType("gamma"));
    JUST(SetParamDataType("mean"));
    JUST(SetParamDataType("inv_variance"));
    if (ctx->has_output("reserve_space", 0)) {
      CHECK_OR_RETURN(reserve_space_infer_fn);
      reserve_space_infer_fn(ctx, &x, ctx->MutOutputTensorDesc("reserve_space", 0));
    }
    return Maybe<void>::Ok();
  };
}

user_op::TensorDescInferFn MakeFwTensorDescInferFn() {
  return MakeFwTensorDescInferFn(
      std::function<Maybe<void>(user_op::InferContext * ctx, const user_op::TensorDesc* x,
                                user_op::TensorDesc* reserve_space)>());
}

user_op::DataTypeInferFn MakeFwDataTypeInferFn() {
  return MakeFwDataTypeInferFn(
      std::function<Maybe<void>(user_op::InferContext * ctx, const user_op::TensorDesc* x,
                                user_op::TensorDesc* reserve_space)>());
}

}  // namespace

/* static */ Maybe<void> NormalizationOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return MakeFwTensorDescInferFn()(ctx);
}

/*static*/ Maybe<void> NormalizationOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> NormalizationOp::GetSbp(user_op::SbpContext* ctx) {
  return FwGetSbpFn(ctx);
}

/* static */ Maybe<void> NormalizationOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return FwInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> NormalizationOp::InferDataType(user_op::InferContext* ctx) {
  return MakeFwDataTypeInferFn()(ctx);
}

/* static */ Maybe<void> NormalizationAddReluOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return MakeFwTensorDescInferFn([](user_op::InferContext* ctx, const user_op::TensorDesc* x,
                                    user_op::TensorDesc* reserve_space) -> Maybe<void> {
    const auto& x_desc = ctx->InputTensorDesc("x", 0);
    size_t reserve_space_bits = x_desc.shape().elem_cnt();
    int64_t parallel_num = ctx->parallel_num();
    if (parallel_num != 1) {
      // There no need to call NdSbp4ArgNameAndIndex when parallel_num = 1 in local.
      const NdSbp& x_nd_sbp = ctx->NdSbp4ArgNameAndIndex("x", 0);
      const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
      int64_t split_num = 1;
      for (int32_t i = 0; i < x_nd_sbp.sbp_parallel_size(); ++i) {
        if (x_nd_sbp.sbp_parallel(i).has_split_parallel()) {
          CHECK_EQ_OR_RETURN(x_nd_sbp.sbp_parallel(i).split_parallel().axis(), 0)
              << "blob x in NormalizationAddReluOp only support B or S(0)";
          split_num *= hierarchy.At(i);
        }
      }
      CHECK_EQ_OR_RETURN(reserve_space_bits % split_num, 0);
      reserve_space_bits = reserve_space_bits / split_num;
    }
    reserve_space->set_shape(Shape({static_cast<int64_t>(RoundUp(reserve_space_bits, 32) / 32)}));
    return Maybe<void>::Ok();
  })(ctx);
}

/* static */ Maybe<void> NormalizationAddReluOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return MakeFwTensorDescInferFn([](user_op::InferContext* ctx, const user_op::TensorDesc* x,
                                    user_op::TensorDesc* reserve_space) -> Maybe<void> {
    const auto& x_desc = ctx->InputTensorDesc("x", 0);
    reserve_space->set_shape(
        Shape({static_cast<int64_t>(RoundUp(x_desc.shape().elem_cnt(), 32) / 32)}));
    return Maybe<void>::Ok();
  })(ctx);
}

/* static */ Maybe<void> NormalizationAddReluOp::GetSbp(user_op::SbpContext* ctx) {
  return FwGetSbpFn(ctx);
}

/* static */ Maybe<void> NormalizationAddReluOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return FwInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> NormalizationAddReluOp::InferDataType(user_op::InferContext* ctx) {
  return MakeFwDataTypeInferFn([](user_op::InferContext* ctx, const user_op::TensorDesc* x,
                                  user_op::TensorDesc* reserve_space) -> Maybe<void> {
    reserve_space->set_data_type(DataType::kInt32);
    return Maybe<void>::Ok();
  })(ctx);
}

#if defined(WITH_CUDA) && (CUDNN_VERSION >= 7401)

namespace {

void InferCudnnReserveSpaceSize(DataType data_type, cudnnBatchNormOps_t ops, int64_t n, int64_t c,
                                int64_t h, int64_t w, size_t* reserve_space_size) {
  cudnnHandle_t cudnn_handle = Singleton<CudnnHandlePool>::Get()->Get();
  CudnnTensorDesc xy_desc(CUDNN_TENSOR_NHWC, data_type, n, c, h, w);
  CudnnActivationDesc activation_desc(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);
  OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      cudnn_handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops, activation_desc.Get(), xy_desc.Get(),
      reserve_space_size));
  Singleton<CudnnHandlePool>::Get()->Put(cudnn_handle);
}

}  // namespace

/* static */ Maybe<void> CudnnFusedNormalizationAddReluOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return MakeFwTensorDescInferFn([](user_op::InferContext* ctx, const user_op::TensorDesc* x,
                                    user_op::TensorDesc* reserve_space) -> Maybe<void> {
    const Shape& x_shape = x->shape();
    const auto axis = ctx->Attr<int32_t>("axis");
    CHECK_EQ_OR_RETURN(x_shape.Count(axis + 1), 1);
    int64_t n = x_shape.At(0);
    {
      const auto& x_nd_sbp = ctx->NdSbp4ArgNameAndIndex("x", 0);
      const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
      int64_t split_num = 1;
      for (int32_t i = 0; i < x_nd_sbp.sbp_parallel_size(); ++i) {
        if (x_nd_sbp.sbp_parallel(i).has_split_parallel()) {
          CHECK_EQ_OR_RETURN(x_nd_sbp.sbp_parallel(i).split_parallel().axis(), 0)
              << "blob x in CudnnFusedNormalizationAddReluOp only support B or S(0)";
          split_num *= hierarchy.At(i);
        }
      }
      CHECK_EQ_OR_RETURN(n % split_num, 0);
      n = n / split_num;
    }
    int64_t h = x_shape.Count(1, axis);
    int64_t w = 1;
    int64_t c = x_shape.At(axis);
    cudnnBatchNormOps_t ops;
    if (ctx->has_input("addend", 0)) {
      ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    } else {
      ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    }
    size_t reserve_space_size;
    InferCudnnReserveSpaceSize(x->data_type(), ops, n, c, h, w, &reserve_space_size);
    reserve_space_size = std::max(reserve_space_size, GetOneVal<size_t>());
    reserve_space->set_shape(Shape({static_cast<int64_t>(reserve_space_size)}));
    return Maybe<void>::Ok();
  })(ctx);
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return MakeFwTensorDescInferFn([](user_op::InferContext* ctx, const user_op::TensorDesc* x,
                                    user_op::TensorDesc* reserve_space) -> Maybe<void> {
    const Shape& x_shape = x->shape();
    const auto axis = ctx->Attr<int32_t>("axis");
    CHECK_EQ_OR_RETURN(x_shape.Count(axis + 1), 1);
    int64_t n = x_shape.At(0);
    int64_t h = x_shape.Count(1, axis);
    int64_t w = 1;
    int64_t c = x_shape.At(axis);
    cudnnBatchNormOps_t ops;
    if (ctx->has_input("addend", 0)) {
      ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    } else {
      ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    }
    size_t reserve_space_size;
    InferCudnnReserveSpaceSize(x->data_type(), ops, n, c, h, w, &reserve_space_size);
    reserve_space_size = std::max(reserve_space_size, GetOneVal<size_t>());
    reserve_space->set_shape(Shape({static_cast<int64_t>(reserve_space_size)}));
    return Maybe<void>::Ok();
  })(ctx);
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluOp::GetSbp(user_op::SbpContext* ctx) {
  return FwGetSbpFn(ctx);
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return FwInputArgModifyFn(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluOp::InferDataType(
    user_op::InferContext* ctx) {
  return MakeFwDataTypeInferFn([](user_op::InferContext* ctx, const user_op::TensorDesc* x,
                                  user_op::TensorDesc* reserve_space) -> Maybe<void> {
    reserve_space->set_data_type(DataType::kChar);
    return Maybe<void>::Ok();
  })(ctx);
}

#else

/* static */ Maybe<void> CudnnFusedNormalizationAddReluOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA and CuDNN >= 7401";
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA and CuDNN >= 7401";
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluOp::GetSbp(user_op::SbpContext* ctx) {
  return Error::UnimplementedError() << "require CUDA and CuDNN >= 7401";
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return Error::UnimplementedError() << "require CUDA and CuDNN >= 7401";
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluOp::InferDataType(
    user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA and CuDNN >= 7401";
}

#endif  // WITH_CUDA

namespace {

Maybe<void> BwTensorDescInferFn(user_op::InferContext* ctx) {
#ifdef WITH_CUDA
  // assume cudnn is enabled
  CHECK_GE_OR_RETURN(ctx->Attr<float>("epsilon"), CUDNN_BN_MIN_EPSILON);
#endif
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const Shape& x_shape = x.shape();
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  CHECK_EQ_OR_RETURN(dy.shape(), x_shape);
  if (ctx->has_input("y", 0)) {
    const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
    CHECK_EQ_OR_RETURN(y.shape(), x_shape);
  }
  *ctx->MutOutputTensorDesc("dx", 0) = x;
  if (ctx->has_output("addend_diff", 0)) { *ctx->MutOutputTensorDesc("addend_diff", 0) = x; }
  const Shape param_shape({x_shape.At(ctx->Attr<int32_t>("axis"))});
  const auto CheckParamTensorDesc = MakeCheckParamTensorDescFn(ctx, param_shape);
  const auto SetParamTensorDesc = MakeSetParamTensorDescFn(ctx, param_shape);
  JUST(CheckParamTensorDesc("mean"));
  JUST(CheckParamTensorDesc("inv_variance"));
  JUST(CheckParamTensorDesc("gamma"));
  JUST(CheckParamTensorDesc("beta"));
  JUST(SetParamTensorDesc("gamma_diff"));
  JUST(SetParamTensorDesc("beta_diff"));
  return Maybe<void>::Ok();
}

Maybe<void> BwDataTypeInferFn(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const DataType x_type = x.data_type();
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  CHECK_EQ_OR_RETURN(dy.data_type(), x_type)
      << "InferDataType Failed. Expected " << DataType_Name(x_type) << ", but got "
      << DataType_Name(dy.data_type());
  if (ctx->has_input("y", 0)) {
    const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
    CHECK_EQ_OR_RETURN(y.data_type(), x_type)
        << "InferDataType Failed. Expected " << DataType_Name(x_type) << ", but got "
        << DataType_Name(y.data_type());
  }
  *ctx->MutOutputTensorDesc("dx", 0) = x;
  if (ctx->has_output("addend_diff", 0)) { *ctx->MutOutputTensorDesc("addend_diff", 0) = x; }
  const DataType param_data_type =
      (x_type == DataType::kFloat16 || x_type == DataType::kBFloat16) ? DataType::kFloat : x_type;
  const auto CheckParamDataType = MakeCheckParamDataTypeFn(ctx, param_data_type);
  const auto SetParamDataType = MakeSetParamDataTypeFn(ctx, param_data_type);
  JUST(CheckParamDataType("mean"));
  JUST(CheckParamDataType("inv_variance"));
  JUST(CheckParamDataType("gamma"));
  JUST(CheckParamDataType("beta"));
  JUST(SetParamDataType("gamma_diff"));
  JUST(SetParamDataType("beta_diff"));
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

}  // namespace

/* static */ Maybe<void> NormalizationGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return BwTensorDescInferFn(ctx);
}

/*static*/ Maybe<void> NormalizationGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> NormalizationGradOp::GetSbp(user_op::SbpContext* ctx) {
  return BwGetSbpFn(ctx);
}

/* static */ Maybe<void> NormalizationGradOp::InferDataType(user_op::InferContext* ctx) {
  return BwDataTypeInferFn(ctx);
}

/* static */ Maybe<void> NormalizationAddReluGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return BwTensorDescInferFn(ctx);
}

/*static*/ Maybe<void> NormalizationAddReluGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> NormalizationAddReluGradOp::GetSbp(user_op::SbpContext* ctx) {
  return BwGetSbpFn(ctx);
}

/* static */ Maybe<void> NormalizationAddReluGradOp::InferDataType(user_op::InferContext* ctx) {
  return BwDataTypeInferFn(ctx);
}

#if defined(WITH_CUDA) && (CUDNN_VERSION >= 7401)

/* static */ Maybe<void> CudnnFusedNormalizationAddReluGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return BwTensorDescInferFn(ctx);
}

/*static*/ Maybe<void> CudnnFusedNormalizationAddReluGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluGradOp::GetSbp(user_op::SbpContext* ctx) {
  return BwGetSbpFn(ctx);
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluGradOp::InferDataType(
    user_op::InferContext* ctx) {
  return BwDataTypeInferFn(ctx);
}

#else

/* static */ Maybe<void> CudnnFusedNormalizationAddReluGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA and CuDNN >= 7401";
}

/*static*/ Maybe<void> CudnnFusedNormalizationAddReluGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA and CuDNN >= 7401";
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluGradOp::GetSbp(user_op::SbpContext* ctx) {
  return Error::UnimplementedError() << "require CUDA and CuDNN >= 7401";
}

/* static */ Maybe<void> CudnnFusedNormalizationAddReluGradOp::InferDataType(
    user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA and CuDNN >= 7401";
}

#endif
}  // namespace oneflow
