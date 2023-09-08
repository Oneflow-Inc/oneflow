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
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<double> GetComputationCost(user_op::ComputeComplexityFnContext* ctx) {
  bool transpose_b = ctx->Attr<bool>("transpose_b");
  const Shape& shape_b = ctx->Shape4ArgNameAndIndex("b", 0);
  int64_t n = 0;
  if (!transpose_b) {
    n = shape_b.At(shape_b.NumAxes() - 1);
  } else {
    n = shape_b.At(shape_b.NumAxes() - 2);
  }

  double logical_computation_cost = 2 * ctx->Shape4ArgNameAndIndex("a", 0).elem_cnt() * n;
  const auto& nd_sbp_a = ctx->NdSbp4ArgNameAndIndex("a", 0);
  const auto& nd_sbp_b = ctx->NdSbp4ArgNameAndIndex("b", 0);
  const auto& parallel_hierarchy = ctx->parallel_desc().hierarchy();
  for (int32_t sbp_dim = 0; sbp_dim < nd_sbp_a.sbp_parallel_size(); sbp_dim++) {
    if (nd_sbp_a.sbp_parallel(sbp_dim).has_split_parallel()
        || nd_sbp_b.sbp_parallel(sbp_dim).has_split_parallel()) {
      logical_computation_cost /= parallel_hierarchy->At(sbp_dim);
    }
  }
  return logical_computation_cost;
}

}  // namespace

/* static */ Maybe<void> GroupedMatmulQuantOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  bool transpose_a = ctx->Attr<bool>("transpose_a");
  bool transpose_b = ctx->Attr<bool>("transpose_b");
  CHECK_EQ_OR_RETURN(transpose_a, false);
  CHECK_EQ_OR_RETURN(transpose_b, true);

  const int64_t input_size = ctx->input_size("as");
  CHECK_EQ_OR_RETURN(ctx->input_size("bs"), input_size);
  const bool has_sacles = ctx->has_input("scales", 0);
  const bool has_biases = ctx->has_input("biases", 0);
  const bool has_in_zero_points = ctx->has_input("in_zero_points", 0);
  const bool has_add_to_outputs = ctx->has_input("_add_to_outputs", 0);
  if (has_in_zero_points) {
    CHECK_EQ_OR_RETURN(has_sacles, false);
    CHECK_EQ_OR_RETURN(ctx->input_size("in_zero_points"), input_size);
    CHECK_OR_RETURN(ctx->has_input("in_scales", 0));
    CHECK_EQ_OR_RETURN(ctx->input_size("in_scales"), input_size);
    CHECK_OR_RETURN(ctx->has_input("weight_scales", 0));
    CHECK_EQ_OR_RETURN(ctx->input_size("weight_scales"), input_size);
    CHECK_OR_RETURN(ctx->has_input("weight_accs", 0));
    CHECK_EQ_OR_RETURN(ctx->input_size("weight_accs"), input_size);
    if (has_biases) { CHECK_EQ_OR_RETURN(ctx->input_size("biases"), input_size); }
  }
  if (has_sacles) {
    CHECK_EQ_OR_RETURN(ctx->input_size("scales"), input_size);
    CHECK_EQ_OR_RETURN(has_biases, true);
    CHECK_EQ_OR_RETURN(ctx->input_size("biases"), input_size);
  }
  if (has_add_to_outputs) { CHECK_EQ_OR_RETURN(ctx->input_size("_add_to_outputs"), input_size); }
  CHECK_EQ_OR_RETURN(ctx->output_size("outputs"), input_size);

  const DataType weight_data_type = ctx->InputTensorDesc("bs", 0).data_type();
  const DataType out_data_type = ctx->Attr<DataType>("out_dtype");
  for (int64_t i = 0; i < input_size; ++i) {
    const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("as", i);
    CHECK_EQ_OR_RETURN(x_desc.data_type(), weight_data_type);
    CHECK_GE_OR_RETURN(x_desc.shape().NumAxes(), 2);
    const int64_t k = x_desc.shape().At(x_desc.shape().NumAxes() - 1);
    const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("bs", i);
    CHECK_EQ_OR_RETURN(weight_desc.data_type(), weight_data_type);
    CHECK_EQ_OR_RETURN(weight_desc.shape().NumAxes(), 2);
    CHECK_EQ_OR_RETURN(weight_desc.shape().At(1), k);
    const int64_t n = weight_desc.shape().At(0);
    if (has_in_zero_points) {
      const user_op::TensorDesc& in_zero_point = ctx->InputTensorDesc("in_zero_points", i);
      CHECK_EQ_OR_RETURN(in_zero_point.data_type(), weight_data_type);
      CHECK_EQ_OR_RETURN(in_zero_point.shape().Count(0), 1);
      const user_op::TensorDesc& in_scale = ctx->InputTensorDesc("in_scales", i);
      CHECK_EQ_OR_RETURN(in_scale.data_type(), DataType::kFloat);
      CHECK_EQ_OR_RETURN(in_scale.shape().Count(0), 1);
      const user_op::TensorDesc& weight_scale = ctx->InputTensorDesc("weight_scales", i);
      CHECK_EQ_OR_RETURN(weight_scale.data_type(), out_data_type);
      CHECK_EQ_OR_RETURN(weight_scale.shape(), Shape({n}));
      const user_op::TensorDesc& weight_acc = ctx->InputTensorDesc("weight_accs", i);
      CHECK_EQ_OR_RETURN(weight_acc.data_type(), out_data_type);
      CHECK_EQ_OR_RETURN(weight_acc.shape(), Shape({n}));
      if (has_biases) {
        const user_op::TensorDesc& bias = ctx->InputTensorDesc("biases", i);
        CHECK_EQ_OR_RETURN(bias.data_type(), out_data_type);
        CHECK_EQ_OR_RETURN(bias.shape(), Shape({n}));
      }
    }
    if (has_sacles) {
      CHECK_OR_RETURN(ctx->has_input("biases", i));
      const user_op::TensorDesc& scale = ctx->InputTensorDesc("scales", i);
      CHECK_EQ_OR_RETURN(scale.data_type(), out_data_type);
      CHECK_EQ_OR_RETURN(scale.shape(), Shape({n}));
      const user_op::TensorDesc& bias = ctx->InputTensorDesc("biases", i);
      CHECK_EQ_OR_RETURN(bias.shape(), Shape({n}));
      CHECK_EQ_OR_RETURN(bias.data_type(), out_data_type);
    }
    user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("outputs", i);
    y_desc->set_data_type(out_data_type);
    DimVector out_dim_vec = x_desc.shape().dim_vec();
    out_dim_vec.back() = n;
    y_desc->set_shape(Shape(out_dim_vec));
    if (has_add_to_outputs) {
      const auto& add_to_output = ctx->InputTensorDesc("_add_to_outputs", i);
      CHECK_EQ_OR_RETURN(add_to_output.data_type(), out_data_type);
      CHECK_EQ_OR_RETURN(add_to_output.shape(), y_desc->shape());
    }
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> GroupedMatmulQuantOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> GroupedMatmulQuantOp::GetSbp(user_op::SbpContext* ctx) {
  {
    // s0 x b
    auto builder = ctx->NewBuilder();
    for (int64_t i = 0; i < ctx->user_op_conf().input_size("as"); ++i) {
      builder.Split(user_op::OpArg("as", i), 0);
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("bs"); ++i) {
      builder.Broadcast(user_op::OpArg("bs", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("in_zero_points"); ++i) {
      builder.Broadcast(user_op::OpArg("in_zero_points", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("in_scales"); ++i) {
      builder.Broadcast(user_op::OpArg("in_scales", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("weight_scales"); ++i) {
      builder.Broadcast(user_op::OpArg("weight_scales", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("weight_accs"); ++i) {
      builder.Broadcast(user_op::OpArg("weight_accs", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("scales"); ++i) {
      builder.Broadcast(user_op::OpArg("scales", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("biases"); ++i) {
      builder.Broadcast(user_op::OpArg("biases", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("_add_to_outputs"); ++i) {
      builder.Split(user_op::OpArg("_add_to_outputs", i), 0);
    }
    for (int i = 0; i < ctx->user_op_conf().output_size("outputs"); ++i) {
      builder.Split(user_op::OpArg("outputs", i), 0);
    }
    builder.Build();
  }

  {
    // b x s0
    auto builder = ctx->NewBuilder();
    for (int64_t i = 0; i < ctx->user_op_conf().input_size("as"); ++i) {
      builder.Broadcast(user_op::OpArg("as", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("bs"); ++i) {
      builder.Split(user_op::OpArg("bs", i), 0);
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("in_zero_points"); ++i) {
      builder.Broadcast(user_op::OpArg("in_zero_points", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("in_scales"); ++i) {
      builder.Broadcast(user_op::OpArg("in_scales", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("weight_scales"); ++i) {
      builder.Split(user_op::OpArg("weight_scales", i), 0);
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("weight_accs"); ++i) {
      builder.Split(user_op::OpArg("weight_accs", i), 0);
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("scales"); ++i) {
      builder.Split(user_op::OpArg("scales", i), 0);
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("biases"); ++i) {
      builder.Split(user_op::OpArg("biases", i), 0);
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("_add_to_outputs"); ++i) {
      builder.Split(user_op::OpArg("_add_to_outputs", i),
                    ctx->LogicalTensorDesc4InputArgNameAndIndex("as", i).shape().NumAxes() - 1);
    }
    for (int i = 0; i < ctx->user_op_conf().output_size("outputs"); ++i) {
      builder.Split(user_op::OpArg("outputs", i),
                    ctx->LogicalTensorDesc4InputArgNameAndIndex("as", i).shape().NumAxes() - 1);
    }
    builder.Build();
  }

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> GroupedMatmulQuantOp::InferDataType(user_op::InferContext* ctx) {
  const DataType out_data_type = ctx->Attr<DataType>("out_dtype");
  for (int32_t i = 0; i < ctx->output_size("outputs"); i++) {
    user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("outputs", i);
    y_desc->set_data_type(out_data_type);
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<double> GroupedMatmulQuantOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return GetComputationCost(ctx);
}

}  // namespace oneflow
