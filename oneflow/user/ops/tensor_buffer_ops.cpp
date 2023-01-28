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

namespace oneflow {

/*static*/ Maybe<void> TensorBufferToTensorOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("in", 0), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TensorBufferToTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_is_dynamic(in.is_dynamic());
  const auto& instance_shape = ctx->Attr<Shape>("instance_shape");
  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), in.shape().dim_vec().cbegin(), in.shape().dim_vec().cend());
  dim_vec.insert(dim_vec.end(), instance_shape.dim_vec().cbegin(), instance_shape.dim_vec().cend());
  out->set_shape(Shape(dim_vec));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TensorBufferToTensorOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TensorBufferToTensorOp::InferDataType(user_op::InferContext* ctx) {
  const auto data_type = ctx->Attr<DataType>("dtype");
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  CHECK_OR_RETURN(IsTriviallyCopyableDataType(data_type));
  out->set_data_type(data_type);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TensorToTensorBufferOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  const auto& instance_dims = ctx->Attr<int32_t>("instance_dims");
  CHECK_LE_OR_RETURN(instance_dims, in.shape().NumAxes());
  FOR_RANGE(int64_t, i, 0, in.shape().NumAxes() - instance_dims) {
    ctx->NewBuilder().Split(user_op::OpArg("in", 0), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TensorToTensorBufferOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  const Shape& in_shape = in.shape();
  const auto& instance_dims = ctx->Attr<int32_t>("instance_dims");
  CHECK_LT_OR_RETURN(instance_dims, in_shape.NumAxes());
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_is_dynamic(in.is_dynamic());
  DimVector out_dim_vec;
  out_dim_vec.insert(out_dim_vec.end(), in_shape.dim_vec().cbegin(),
                     in_shape.dim_vec().cend() - instance_dims);
  out->set_shape(Shape(out_dim_vec));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TensorToTensorBufferOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TensorToTensorBufferOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_OR_RETURN(IsTriviallyCopyableDataType(in.data_type()));
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_data_type(DataType::kTensorBuffer);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> GenTensorBufferOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> GenTensorBufferOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  const Shape& shape = ctx->Attr<Shape>("shape");
  const int64_t num_tensor_buffers = shape.elem_cnt();
  const std::vector<Shape>& shape_list = ctx->Attr<std::vector<Shape>>("shape_list");
  const std::vector<float>& value_list = ctx->Attr<std::vector<float>>("value_list");
  CHECK_EQ_OR_RETURN(num_tensor_buffers, shape_list.size());
  CHECK_EQ_OR_RETURN(num_tensor_buffers, value_list.size());
  out->set_shape(shape);
  out->set_is_dynamic(ctx->Attr<bool>("dynamic_out"));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> GenTensorBufferOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> GenTensorBufferOp::InferDataType(user_op::InferContext* ctx) {
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_data_type(DataType::kTensorBuffer);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TensorBufferToListOfTensorsOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> TensorBufferToListOfTensorsOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_GT_OR_RETURN(in.shape().elem_cnt(), 0);
  CHECK_OR_RETURN(!in.is_dynamic());
  const Shape& out_shape = ctx->Attr<Shape>("out_shape");
  const bool dynamic_out = ctx->Attr<bool>("dynamic_out");
  int64_t num_tensor_buffers = in.shape().elem_cnt();
  for (int64_t i = 0; i < num_tensor_buffers; ++i) {
    user_op::TensorDesc* out_i = ctx->MutOutputTensorDesc("out", i);
    out_i->set_shape(out_shape);
    out_i->set_is_dynamic(dynamic_out);
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TensorBufferToListOfTensorsOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TensorBufferToListOfTensorsOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_EQ_OR_RETURN(in.data_type(), DataType::kTensorBuffer)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kTensorBuffer) << ", but got "
      << DataType_Name(in.data_type());
  const DataType out_dtype = ctx->Attr<DataType>("out_dtype");
  CHECK_OR_RETURN(IsTriviallyCopyableDataType(out_dtype));
  int64_t num_tensor_buffers = ctx->outputs().size();
  for (int64_t i = 0; i < num_tensor_buffers; ++i) {
    user_op::TensorDesc* out_i = ctx->MutOutputTensorDesc("out", i);
    out_i->set_data_type(out_dtype);
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TensorBufferToListOfTensorsOp::ModifyOutputArg(
    const GetOutputArgModifier& GetOutputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  if (conf.attr<bool>("dynamic_out")) {
    FOR_RANGE(int64_t, i, 0, conf.output_size("out")) {
      user_op::OutputArgModifier* out_i_modifier = GetOutputArgModifierFn("out", i);
      CHECK_OR_RETURN(out_i_modifier != nullptr);
      out_i_modifier->set_header_infered_before_compute(false);
    }
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TensorBufferToListOfTensorsOp::CheckAttr(
    const user_op::UserOpDefWrapper&, const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.output_size("out") >= 1);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TensorBufferToListOfTensorsV2Op::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> TensorBufferToListOfTensorsV2Op::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_GT_OR_RETURN(in.shape().elem_cnt(), 0);
  CHECK_OR_RETURN(!in.is_dynamic());
  const std::vector<Shape>& out_shapes = ctx->Attr<std::vector<Shape>>("out_shapes");
  const bool dynamic_out = ctx->Attr<bool>("dynamic_out");
  int64_t num_tensor_buffers = in.shape().elem_cnt();
  for (int64_t i = 0; i < num_tensor_buffers; ++i) {
    user_op::TensorDesc* out_i = ctx->MutOutputTensorDesc("out", i);
    out_i->set_shape(out_shapes[i]);
    out_i->set_is_dynamic(dynamic_out);
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TensorBufferToListOfTensorsV2Op::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TensorBufferToListOfTensorsV2Op::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_EQ_OR_RETURN(in.data_type(), DataType::kTensorBuffer)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kTensorBuffer) << ", but got "
      << DataType_Name(in.data_type());
  const std::vector<DataType>& out_dtypes = ctx->Attr<std::vector<DataType>>("out_dtypes");
  int64_t num_tensor_buffers = ctx->outputs().size();
  for (int64_t i = 0; i < num_tensor_buffers; ++i) {
    CHECK_OR_RETURN(IsTriviallyCopyableDataType(out_dtypes[i]));
    user_op::TensorDesc* out_i = ctx->MutOutputTensorDesc("out", i);
    out_i->set_data_type(out_dtypes[i]);
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TensorBufferToListOfTensorsV2Op::ModifyOutputArg(
    const GetOutputArgModifier& GetOutputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  if (conf.attr<bool>("dynamic_out")) {
    FOR_RANGE(int64_t, i, 0, conf.output_size("out")) {
      user_op::OutputArgModifier* out_i_modifier = GetOutputArgModifierFn("out", i);
      CHECK_OR_RETURN(out_i_modifier != nullptr);
      out_i_modifier->set_header_infered_before_compute(false);
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TensorBufferToListOfTensorsV2Op::CheckAttr(
    const user_op::UserOpDefWrapper&, const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.output_size("out") >= 1);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
