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

REGISTER_CPU_ONLY_USER_OP("image_decode")
    .Input("in")
    .Output("out")
    .Attr<std::string>("color_space", "BGR")
    .Attr<DataType>("data_type", DataType::kUInt8)
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& def,
                       const user_op::UserOpConfWrapper& conf) -> Maybe<void> {
      bool check_failed = false;
      std::stringstream err;
      err << "Illegal attr value for " << conf.op_type_name() << " op, op_name: " << conf.op_name();
      const std::string& color_space = conf.attr<std::string>("color_space");
      if (color_space != "BGR" && color_space != "RGB" && color_space != "GRAY") {
        err << ", color_space: " << color_space
            << " (color_space can only be one of BGR, RGB and GRAY)";
        check_failed = true;
      }
      DataType data_type = conf.attr<DataType>("data_type");
      if (data_type != DataType::kUInt8 && data_type != DataType::kFloat) {
        err << ", data_type: " << data_type << " (only support kUInt8 and kFloat for now)";
        check_failed = true;
      }
      if (check_failed) { return oneflow::Error::CheckFailedError() << err.str(); }
      return Maybe<void>::Ok();
    })
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      CHECK_OR_RETURN(in_desc->data_type() == DataType::kTensorBuffer);
      CHECK_OR_RETURN(in_desc->shape().NumAxes() == 1 && in_desc->shape().At(0) >= 1);
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc->mut_shape() = in_desc->shape();
      *out_desc->mut_data_type() = DataType::kTensorBuffer;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
