// /*
// Copyright 2020 The OneFlow Authors. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// */
// #include "oneflow/core/framework/framework.h"
// #include "oneflow/user/kernels/avg_pooling_kernel_util.h"

// namespace oneflow {

// namespace {

// typedef std::function<Maybe<void>(user_op::InferContext* ctx)> TensorDescInferFn;
// typedef std::function<Maybe<void>(const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp)>
//     GenBackwardOpConfFn;

// TensorDescInferFn MakeForwardTensorDescInferFn(const int32_t dim) {
//   return [dim](user_op::InferContext* ctx) -> Maybe<void> {
//     const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
//     const std::string& data_format = ctx->Attr<std::string>("data_format");
//     const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
//     const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
//     const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
//     const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
//     const bool count_include_pad = ctx->Attr<bool>("count_include_pad");
//     const int64_t& divisor_override = ctx->Attr<int64_t>("divisor_override");

//     CHECK_EQ_OR_RETURN(kernel_size.size(), dim);
//     for (int32_t pool_dim : kernel_size) { CHECK_GT_OR_RETURN(pool_dim, 0); }
//     CHECK_EQ_OR_RETURN(stride.size(), dim);
//     for (int32_t stride_dim : stride) { CHECK_GT_OR_RETURN(stride_dim, 0); }
//     for (int32_t i = 0; i < padding.size(); i++) {
//       CHECK_GE_OR_RETURN(kernel_size[i], 2 * padding[i])
//           << "pad should be smaller than half of kernel size";
//     }

//     const AvgPoolingParams3D params_3d(dim, *x_shape, data_format, padding, kernel_size, stride, ceil_mode, count_include_pad, divisor_override);
//     user_op::TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("y", 0);
//     *y_desc = *ctx->TensorDesc4ArgNameAndIndex("x", 0);
//     *y_desc->mut_shape() = params_3d.GetYShape();

//     return Maybe<void>::Ok();
//   };
// }

// Maybe<void> ForwardGetSbpFn(user_op::SbpContext* ctx) {
//   const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
//   const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
//   FOR_RANGE(int64_t, i, 0, std::min(2, (int)tensor.shape().NumAxes())) {
//     if (padding[i] == 0) {
//       ctx->NewBuilder()
//           .Split(user_op::OpArg("x", 0), i)
//           .Split(user_op::OpArg("y", 0), i)
//           .Build();
//     }
//   }
//   return Maybe<void>::Ok();
// }

// Maybe<void> BackwardTensorDescInferFn(user_op::InferContext* ctx) {
//   *ctx->TensorDesc4ArgNameAndIndex("dx", 0) = *ctx->TensorDesc4ArgNameAndIndex("x", 0);
//   return Maybe<void>::Ok();
// }

// Maybe<void> BackwardGetSbpFn(user_op::SbpContext* ctx) {
//   const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
//   const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
//   FOR_RANGE(int64_t, i, 0, std::min(2, (int)tensor.shape().NumAxes())) {
//     if (padding[i] == 0) {
//       ctx->NewBuilder()
//           .Split(user_op::OpArg("x", 0), i)
//           .Split(user_op::OpArg("y", 0), i)
//           .Split(user_op::OpArg("dy", 0), i)
//           .Split(user_op::OpArg("dx", 0), i)
//           .Build();
//     }
//   }
//   return Maybe<void>::Ok();
// }

// Maybe<void> FwInferDataType(user_op::InferContext* ctx) {
//   *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
//   return Maybe<void>::Ok();
// }

// Maybe<void> BwInferDataType(user_op::InferContext* ctx) {
//   *ctx->OutputDType("dx", 0) = ctx->InputDType("x", 0);
//   return Maybe<void>::Ok();
// }

// }  // namespace

// #define REGISTER_AVGPOOL_FORWARD_OP(name, ndim) \
//   REGISTER_USER_OP(name) \
//     .Input("x") \
//     .Output("y") \
//     .Attr<std::vector<int32_t>>("padding") \
//     .Attr<std::string>("data_format") \
//     .Attr<std::vector<int32_t>>("kernel_size") \
//     .Attr<std::vector<int32_t>>("stride") \
//     .Attr<bool>("ceil_mode") \
//     .Attr<bool>("count_include_pad") \
//     .Attr<int64_t>("divisor_override") \
//     .SetTensorDescInferFn(MakeForwardTensorDescInferFn(ndim)) \
//     .SetGetSbpFn(ForwardGetSbpFn) \
//     .SetDataTypeInferFn(FwInferDataType);


// // REGISTER_AVGPOOL_FORWARD_OP("avgpool_1d", 1);
// REGISTER_AVGPOOL_FORWARD_OP("avgpool_2d", 2);
// // REGISTER_AVGPOOL_FORWARD_OP("avgpool_3d", 3);


// }  // namespace oneflow
