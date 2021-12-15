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

Maybe<void> InferNmsTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = Shape({ctx->InputShape("in", 0).At(0)});
  return Maybe<void>::Ok();
}

Maybe<void> InferNmsDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = DataType::kInt8;
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("nms")
    .Input("in")
    .Output("out")
    .Attr<float>("iou_threshold")
    .Attr<int32_t>("keep_n")
    .SetTensorDescInferFn(InferNmsTensorDesc)
    .SetDataTypeInferFn(InferNmsDataType)
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

}  // namespace oneflow
