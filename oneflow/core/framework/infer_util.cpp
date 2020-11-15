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
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/attr_value_accessor.h"

namespace oneflow {

namespace user_op {

Maybe<void> TensorDescInferFnUtil::Unchanged(InferContext* ctx) {
  const TensorDesc* tensor_desc = nullptr;
  for (size_t i = 0; i < ctx->inputs().size(); ++i) {
    const std::pair<std::string, int32_t>& input_arg = ctx->inputs().at(i);
    if (tensor_desc) {
      CHECK_OR_RETURN(*tensor_desc
                      == *ctx->TensorDesc4ArgNameAndIndex(input_arg.first, input_arg.second));
    } else {
      tensor_desc = ctx->TensorDesc4ArgNameAndIndex(input_arg.first, input_arg.second);
    }
  }
  for (size_t i = 0; i < ctx->outputs().size(); ++i) {
    const std::pair<std::string, int32_t>& output_arg = ctx->outputs().at(i);
    *ctx->TensorDesc4ArgNameAndIndex(output_arg.first, output_arg.second) = *tensor_desc;
  }
  return Maybe<void>::Ok();
}

Maybe<void> TensorDescInferFnUtil::InOutCorrespond(InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->inputs().size(), ctx->outputs().size());
  for (size_t i = 0; i < ctx->inputs().size(); ++i) {
    const auto& input_arg = ctx->inputs().at(i);
    const auto& output_arg = ctx->outputs().at(i);
    *ctx->TensorDesc4ArgNameAndIndex(output_arg.first, output_arg.second) =
        *ctx->TensorDesc4ArgNameAndIndex(input_arg.first, input_arg.second);
  }
  return Maybe<void>::Ok();
}

Maybe<void> CheckAttrFnUtil::NoCheck(const UserOpDefWrapper&, const UserOpConfWrapper&) {
  return Maybe<void>::Ok();
}

size_t TmpSizeInferFnUtil::ZeroTmpSize(InferContext*) { return 0; }

}  // namespace user_op

}  // namespace oneflow
