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
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

namespace {
Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  const auto& reduce_axes = ctx->Attr<std::vector<int32_t>>("axis");
  CHECK_OR_RETURN(!reduce_axes.empty());
  const AxisVector reduce_axes_vec = {reduce_axes.begin(), reduce_axes.end()};
  const Shape& reduce_shape = CreateReducedShape(input_shape, reduce_axes_vec);
  const bool keepdim = ctx->Attr<bool>("keepdim");
  Shape* output_shape = ctx->OutputShape("output", 0);
  if (keepdim) {
    *output_shape = reduce_shape;
  } else {
    *output_shape = reduce_shape.RemoveOnes(reduce_axes_vec);
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("output", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpFn(user_op::SbpContext* ctx) { return Maybe<void>::Ok(); }
}  // namespace

REGISTER_USER_OP("var")
    .Input("input")
    .Output("output")
    .Attr<std::vector<int32_t>>("axis")
    .Attr<bool>("unbiased", true)
    .Attr<bool>("keepdim", false)
    .SetTensorDescInferFn(InferTensorDescFn)
    .SetGetSbpFn(GetSbpFn)
    .SetDataTypeInferFn(InferDataType);

}  // namespace oneflow
