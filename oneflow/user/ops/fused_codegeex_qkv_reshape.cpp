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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

Maybe<void> FusedCodegeexQkvReshapeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& query = ctx->InputTensorDesc("query", 0);
  const user_op::TensorDesc& key = ctx->InputTensorDesc("key", 0);
  const user_op::TensorDesc& value = ctx->InputTensorDesc("value", 0);
  const int32_t num_attention_heads = ctx->Attr<int32_t>("num_attention_heads");
  CHECK_EQ_OR_RETURN(query.shape().size(), 3) << "query shape size should be equal 3";
  CHECK_EQ_OR_RETURN(key.shape().size(), 3) << "key shape size should be equal 3";
  CHECK_EQ_OR_RETURN(value.shape().size(), 3) << "value shape size should be equal 3";
  CHECK_EQ_OR_RETURN(query.shape(), key.shape())
      << "query, key, value should has same shape in codegeex attention block";
  CHECK_EQ_OR_RETURN(query.shape(), value.shape())
      << "query, key, value should has same shape in codegeex attention block";
  CHECK_EQ_OR_RETURN(query.shape()[2] % num_attention_heads, 0)
      << "hidden_size must be divisible by num_attention_heads";

  Shape new_shape(DimVector{query.shape()[0], query.shape()[1], num_attention_heads,
                            query.shape()[2] / num_attention_heads});
  user_op::TensorDesc* new_query = ctx->MutOutputTensorDesc("new_query", 0);
  new_query->set_is_dynamic(query.is_dynamic());
  new_query->set_shape(new_shape);

  user_op::TensorDesc* new_key = ctx->MutOutputTensorDesc("new_key", 0);
  new_key->set_is_dynamic(key.is_dynamic());
  new_key->set_shape(new_shape);

  user_op::TensorDesc* new_value = ctx->MutOutputTensorDesc("new_value", 0);
  new_value->set_is_dynamic(value.is_dynamic());
  new_value->set_shape(new_shape);

  return Maybe<void>::Ok();
}

Maybe<void> FusedCodegeexQkvReshapeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return FusedCodegeexQkvReshapeOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> FusedCodegeexQkvReshapeOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& query = ctx->InputTensorDesc("query", 0);
  const user_op::TensorDesc& key = ctx->InputTensorDesc("key", 0);
  const user_op::TensorDesc& value = ctx->InputTensorDesc("value", 0);

  user_op::TensorDesc* new_query = ctx->MutOutputTensorDesc("new_query", 0);
  new_query->set_data_type(query.data_type());
  user_op::TensorDesc* new_key = ctx->MutOutputTensorDesc("new_key", 0);
  new_key->set_data_type(key.data_type());
  user_op::TensorDesc* new_value = ctx->MutOutputTensorDesc("new_value", 0);
  new_value->set_data_type(value.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> FusedCodegeexQkvReshapeOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& query = ctx->LogicalTensorDesc4InputArgNameAndIndex("query", 0);
  FOR_RANGE(int64_t, i, 0, query.shape().NumAxes() - 1) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("query", 0), i)
        .Split(user_op::OpArg("key", 0), i)
        .Split(user_op::OpArg("value", 0), i)
        .Split(user_op::OpArg("new_query", 0), i)
        .Split(user_op::OpArg("new_key", 0), i)
        .Split(user_op::OpArg("new_value", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
