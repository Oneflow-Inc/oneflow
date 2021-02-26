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

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const Shape* input_shape = ctx->Shape4ArgNameAndIndex("input", 0);
  const DataType dtype = ctx->Attr<DataType>("dtype");
  user_op::TensorDesc* output_desc = ctx->TensorDesc4ArgNameAndIndex("output", 0);
  *output_desc->mut_shape() = Shape({input_shape->elem_cnt(), input_shape->NumAxes()});
  *output_desc->mut_data_type() = dtype;
  output_desc->set_is_dynamic(true);
  user_op::TensorDesc* output_size_desc = ctx->TensorDesc4ArgNameAndIndex("output_size", 0);
  *output_size_desc->mut_shape() = Shape({1});
  *output_size_desc->mut_data_type() = dtype;
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("argwhere")
    .Input("input")
    .Output("output")
    .Output("output_size")
    .Attr<DataType>("dtype", DataType::kInt32)
    .SetTensorDescInferFn(InferTensorDesc);

}  // namespace oneflow
