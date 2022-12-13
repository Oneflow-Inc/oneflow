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

/* static */ Maybe<void> RawReaderOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& instance_shape = ctx->Attr<Shape>("shape");
  const int32_t batch_size = ctx->Attr<int64_t>("batch_size");
  DimVector dim_vec;
  dim_vec.push_back(batch_size);
  for (int64_t i = 0; i < instance_shape.NumAxes(); ++i) {
    dim_vec.push_back(instance_shape.At(i));
  }
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  out_tensor->set_shape(Shape(dim_vec));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RawReaderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out", 0);
  int32_t batch_size = ctx->Attr<int64_t>("batch_size");
  int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  if (parallel_num > 1) {
    int64_t split_num = 1;
    const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
    const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
    for (int32_t i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
      if (nd_sbp.sbp_parallel(i).has_split_parallel()) { split_num *= hierarchy.At(i); }
    }
    CHECK_EQ_OR_RETURN(batch_size % split_num, 0) << "batch_size must be a multiple of shard num";
    batch_size /= split_num;
  }
  const Shape& instance_shape = ctx->Attr<Shape>("shape");
  DimVector dim_vec;
  dim_vec.push_back(batch_size);
  for (int64_t i = 0; i < instance_shape.NumAxes(); ++i) {
    dim_vec.push_back(instance_shape.At(i));
  }
  out_tensor->set_shape(Shape({dim_vec}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RawReaderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> RawReaderOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  SbpParallel default_sbp;
  default_sbp.mutable_split_parallel()->set_axis(0);
  return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
}

/* static */ Maybe<void> RawReaderOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->Attr<DataType>("data_type"));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
