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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/user/image/image_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/job/nd_sbp_util.h"

namespace oneflow {

/* static */ Maybe<void> CropMirrorNormalizeFromTensorbufferOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  bool has_mirror = ctx->has_input("mirror", 0);
  if (has_mirror) {
    const user_op::TensorDesc& mirror_tensor = ctx->InputTensorDesc("mirror", 0);
    CHECK_OR_RETURN(mirror_tensor.shape().NumAxes() == 1
                    && in_tensor.shape().At(0) == mirror_tensor.shape().At(0));
  }
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  int64_t N = in_tensor.shape().At(0);
  int64_t H = ctx->Attr<int64_t>("crop_h");
  int64_t W = ctx->Attr<int64_t>("crop_w");
  std::string color_space = ctx->Attr<std::string>("color_space");
  int64_t C = ImageUtil::IsColor(color_space) ? 3 : 1;

  CHECK_OR_RETURN(H != 0 && W != 0);
  CHECK_OR_RETURN(in_tensor.shape().NumAxes() == 1);
  std::string output_layout = ctx->Attr<std::string>("output_layout");
  if (output_layout == "NCHW") {
    out_tensor->set_shape(Shape({N, C, H, W}));
  } else if (output_layout == "NHWC") {
    out_tensor->set_shape(Shape({N, H, W, C}));
  } else {
    return Error::CheckFailedError() << "output_layout: " << output_layout << " is not supported";
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CropMirrorNormalizeFromTensorbufferOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CropMirrorNormalizeFromTensorbufferOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CropMirrorNormalizeFromTensorbufferOp::InferDataType(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  CHECK_EQ_OR_RETURN(in_tensor.data_type(), DataType::kTensorBuffer)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kTensorBuffer) << ", but got "
      << DataType_Name(in_tensor.data_type());
  bool has_mirror = ctx->has_input("mirror", 0);
  if (has_mirror) {
    const user_op::TensorDesc& mirror_tensor = ctx->InputTensorDesc("mirror", 0);
    CHECK_EQ_OR_RETURN(mirror_tensor.data_type(), DataType::kInt8)
        << "InferDataType Failed. Expected " << DataType_Name(DataType::kInt8) << ", but got "
        << DataType_Name(mirror_tensor.data_type());
  }

  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  DataType output_dtype = ctx->Attr<DataType>("output_dtype");
  CHECK_EQ_OR_RETURN(output_dtype,
                     DataType::kFloat)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kFloat) << ", but got "
      << DataType_Name(output_dtype);  // only support float now; for float16 in future
  out_tensor->set_data_type(output_dtype);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CropMirrorNormalizeFromUint8Op::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  bool has_mirror = ctx->has_input("mirror", 0);
  if (has_mirror) {
    const user_op::TensorDesc& mirror_tensor = ctx->InputTensorDesc("mirror", 0);
    CHECK_OR_RETURN(mirror_tensor.shape().NumAxes() == 1
                    && in_tensor.shape().At(0) == mirror_tensor.shape().At(0));
  }
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  int64_t N = in_tensor.shape().At(0);
  int64_t H = ctx->Attr<int64_t>("crop_h");
  int64_t W = ctx->Attr<int64_t>("crop_w");
  std::string color_space = ctx->Attr<std::string>("color_space");
  int64_t C = ImageUtil::IsColor(color_space) ? 3 : 1;
  CHECK_EQ_OR_RETURN(in_tensor.shape().NumAxes(), 4);  // {N, H, W, C}
  CHECK_EQ_OR_RETURN(in_tensor.shape().At(3), C);
  if (H == 0 || W == 0) {
    H = in_tensor.shape().At(1);
    W = in_tensor.shape().At(2);
  } else {
    H = std::min(H, in_tensor.shape().At(1));
    W = std::min(W, in_tensor.shape().At(2));
  }
  std::string output_layout = ctx->Attr<std::string>("output_layout");
  if (output_layout == "NCHW") {
    out_tensor->set_shape(Shape({N, C, H, W}));
  } else if (output_layout == "NHWC") {
    out_tensor->set_shape(Shape({N, H, W, C}));
  } else {
    return Error::CheckFailedError() << "output_layout: " << output_layout << " is not supported";
  }

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CropMirrorNormalizeFromUint8Op::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CropMirrorNormalizeFromUint8Op::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CropMirrorNormalizeFromUint8Op::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  CHECK_EQ_OR_RETURN(in_tensor.data_type(), DataType::kUInt8)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kUInt8) << ", but got "
      << DataType_Name(in_tensor.data_type());
  bool has_mirror = ctx->has_input("mirror", 0);
  if (has_mirror) {
    const user_op::TensorDesc& mirror_tensor = ctx->InputTensorDesc("mirror", 0);
    CHECK_EQ_OR_RETURN(mirror_tensor.data_type(), DataType::kInt8)
        << "InferDataType Failed. Expected " << DataType_Name(DataType::kInt8) << ", but got "
        << DataType_Name(mirror_tensor.data_type());
  }
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  DataType output_dtype = ctx->Attr<DataType>("output_dtype");
  CHECK_EQ_OR_RETURN(output_dtype,
                     DataType::kFloat)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kFloat) << ", but got "
      << DataType_Name(output_dtype);  // only support float now; for float16 in future
  out_tensor->set_data_type(output_dtype);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CoinFlipOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  int64_t batch_size = ctx->Attr<int64_t>("batch_size");
  out_tensor->set_shape(Shape({batch_size}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CoinFlipOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& parallel_hierarchy = *ctx->parallel_desc().hierarchy();
  const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  int64_t batch_size = ctx->Attr<int64_t>("batch_size");
  const Shape logical_shape = Shape({batch_size});
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();

  const auto tensor_slice_view =
      GetTensorSliceView4ParallelId(parallel_hierarchy, nd_sbp, logical_shape, parallel_id);
  const Shape& physical_shape = tensor_slice_view.shape();
  ctx->SetOutputShape("out", 0, physical_shape);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CoinFlipOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(user_op::OpArg("out", 0), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CoinFlipOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const Shape& hierarchy = ctx->parallel_hierarchy();
  NdSbp* output_dist = ctx->NdSbp4ArgNameAndIndex("out", 0);
  // the input may be produced by tick which should be broadcast parallel dist
  std::vector<NdSbp*> inputs_dist;
  for (const auto& arg_pair : ctx->inputs()) {
    inputs_dist.emplace_back(ctx->NdSbp4ArgNameAndIndex(arg_pair.first, arg_pair.second));
  }
  const auto& dist_conf = ctx->user_op_conf().attr<std::vector<std::string>>("nd_sbp");
  if (dist_conf.size() == 0) {
    FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
      output_dist->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
      for (auto* input_dist : inputs_dist) {
        input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
      }
    }
  } else {
    CHECK_EQ_OR_RETURN(dist_conf.size(), hierarchy.NumAxes());
    for (const std::string& sbp_str : dist_conf) {
      SbpParallel sbp_parallel;
      CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, &sbp_parallel));
      CHECK_OR_RETURN(
          (sbp_parallel.has_split_parallel() && sbp_parallel.split_parallel().axis() == 0)
          || sbp_parallel.has_broadcast_parallel());
      *output_dist->add_sbp_parallel() = sbp_parallel;
      for (auto* input_dist : inputs_dist) {
        input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
      }
    }
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CoinFlipOp::InferDataType(user_op::InferContext* ctx) {
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  out_tensor->set_data_type(DataType::kInt8);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ImageRandomCropOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  out_tensor->set_shape(in_tensor.shape());
  out_tensor->set_is_dynamic(in_tensor.is_dynamic());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ImageRandomCropOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ImageRandomCropOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);
}

/* static */ Maybe<void> ImageRandomCropOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
  CHECK_NOTNULL_OR_RETURN(in_modifier);
  in_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ImageRandomCropOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  CHECK_OR_RETURN(in_tensor.data_type() == DataType::kTensorBuffer);
  ctx->SetOutputDType("out", 0, in_tensor.data_type());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
