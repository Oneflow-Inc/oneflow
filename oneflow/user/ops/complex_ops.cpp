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
#include <map>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

static std::map<DataType, DataType> complex_to_real_map{{DataType::kComplex32, DataType::kFloat16},
                                                        {DataType::kComplex64, DataType::kFloat},
                                                        {DataType::kComplex128, DataType::kDouble}};
static std::map<DataType, DataType> real_to_complex_map{{DataType::kFloat16, DataType::kComplex32},
                                                        {DataType::kFloat, DataType::kComplex64},
                                                        {DataType::kDouble, DataType::kComplex128}};

/*static*/ Maybe<void> RealOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);
}
/*static*/ Maybe<void> RealOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::Unchanged(ctx);
}
/*static*/ Maybe<void> RealOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> RealOp::InferDataType(user_op::InferContext* ctx) {
  const std::pair<std::string, int32_t>& input_arg = ctx->inputs().at(0);
  const user_op::TensorDesc& tensor_desc = ctx->InputTensorDesc(input_arg.first, input_arg.second);
  const std::pair<std::string, int32_t>& output_arg = ctx->outputs().at(0);
  ctx->SetOutputDType(output_arg.first, output_arg.second,
                      complex_to_real_map[tensor_desc.data_type()]);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RealGradOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);
}
/*static*/ Maybe<void> RealGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::Unchanged(ctx);
}
/*static*/ Maybe<void> RealGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> RealGradOp::InferDataType(user_op::InferContext* ctx) {
  const std::pair<std::string, int32_t>& input_arg = ctx->inputs().at(0);
  const user_op::TensorDesc& tensor_desc = ctx->InputTensorDesc(input_arg.first, input_arg.second);
  const std::pair<std::string, int32_t>& output_arg = ctx->outputs().at(0);
  ctx->SetOutputDType(output_arg.first, output_arg.second,
                      real_to_complex_map[tensor_desc.data_type()]);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ImagOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);
}
/*static*/ Maybe<void> ImagOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::Unchanged(ctx);
}
/*static*/ Maybe<void> ImagOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ImagOp::InferDataType(user_op::InferContext* ctx) {
  const std::pair<std::string, int32_t>& input_arg = ctx->inputs().at(0);
  const user_op::TensorDesc& tensor_desc = ctx->InputTensorDesc(input_arg.first, input_arg.second);
  const std::pair<std::string, int32_t>& output_arg = ctx->outputs().at(0);
  ctx->SetOutputDType(output_arg.first, output_arg.second,
                      complex_to_real_map[tensor_desc.data_type()]);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ImagGradOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);
}
/*static*/ Maybe<void> ImagGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::Unchanged(ctx);
}
/*static*/ Maybe<void> ImagGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ImagGradOp::InferDataType(user_op::InferContext* ctx) {
  const std::pair<std::string, int32_t>& input_arg = ctx->inputs().at(0);
  const user_op::TensorDesc& tensor_desc = ctx->InputTensorDesc(input_arg.first, input_arg.second);
  const std::pair<std::string, int32_t>& output_arg = ctx->outputs().at(0);
  ctx->SetOutputDType(output_arg.first, output_arg.second,
                      real_to_complex_map[tensor_desc.data_type()]);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ConjPhysicalOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);
}
/*static*/ Maybe<void> ConjPhysicalOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::Unchanged(ctx);
}
/*static*/ Maybe<void> ConjPhysicalOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ConjPhysicalOp::InferDataType(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::UnchangedDataType(ctx);
}

}  // namespace oneflow
