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
#include "oneflow/core/operator/operator.h"

namespace oneflow {

REGISTER_USER_OP("_nccl_logical_all_reduce")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *ctx->TensorDesc4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetInferSbpSignatureFn([](user_op::InferSbpSignatureFnContext* ctx) -> Maybe<void> {
      // P2B
      auto* bn2sbp = ctx->mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel();
      const SbpParallel& in_sbp_hint = ctx->SbpParallelHint4InputArgNameAndIndex("in", 0);
      CHECK(in_sbp_hint.has_partial_sum_parallel());
      const std::string& ibn = GenRepeatedBn("in", 0);
      const std::string& obn = GenRepeatedBn("out", 0);
      (*bn2sbp)[ibn].mutable_partial_sum_parallel();
      (*bn2sbp)[obn].mutable_broadcast_parallel();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("_nccl_logical_reduce_scatter")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      const user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      *out_tensor = *in_tensor;
      const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
      Shape* out_shape = out_tensor->mut_shape();
      const Shape& in_shape = in_tensor->shape();
      CHECK_GT(in_shape.NumAxes(), 0);
      CHECK_GT(in_shape.elem_cnt(), 0);
      CHECK_EQ(in_shape.At(0) % parallel_num, 0);
      out_shape->Set(0, in_shape.At(0) / parallel_num);
      return Maybe<void>::Ok();
    })
    .SetInferSbpSignatureFn([](user_op::InferSbpSignatureFnContext* ctx) -> Maybe<void> {
      // P2S
      auto* bn2sbp = ctx->mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel();
      const SbpParallel& in_sbp_hint = ctx->SbpParallelHint4InputArgNameAndIndex("in", 0);
      CHECK(in_sbp_hint.has_partial_sum_parallel());
      const std::string& ibn = GenRepeatedBn("in", 0);
      const std::string& obn = GenRepeatedBn("out", 0);
      (*bn2sbp)[ibn].mutable_partial_sum_parallel();
      (*bn2sbp)[obn].mutable_split_parallel()->set_axis(0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("_nccl_logical_all_gather")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      const user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      *out_tensor = *in_tensor;
      const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
      Shape* out_shape = out_tensor->mut_shape();
      const Shape& in_shape = in_tensor->shape();
      CHECK_GT(in_shape.NumAxes(), 0);
      CHECK_GT(in_shape.elem_cnt(), 0);
      out_shape->Set(0, in_shape.At(0) * parallel_num);
      return Maybe<void>::Ok();
    })
    .SetInferSbpSignatureFn([](user_op::InferSbpSignatureFnContext* ctx) -> Maybe<void> {
      // S2B
      auto* bn2sbp = ctx->mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel();
      const SbpParallel& in_sbp_hint = ctx->SbpParallelHint4InputArgNameAndIndex("in", 0);
      CHECK(in_sbp_hint.has_split_parallel());
      CHECK_EQ(in_sbp_hint.split_parallel().axis(), 0);
      const std::string& ibn = GenRepeatedBn("in", 0);
      const std::string& obn = GenRepeatedBn("out", 0);
      (*bn2sbp)[ibn].mutable_split_parallel()->set_axis(0);
      (*bn2sbp)[obn].mutable_broadcast_parallel();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("_nccl_logical_s2s")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("in_split_axis", -1)
    .Attr<int64_t>("out_split_axis", -1)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      const user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      *out_tensor = *in_tensor;
      const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
      const int64_t in_split_axis = ctx->Attr<int64_t>("in_split_axis");
      const int64_t out_split_axis = ctx->Attr<int64_t>("out_split_axis");
      CHECK_NE(in_split_axis, out_split_axis);
      CHECK_NE(in_split_axis, -1);
      CHECK_NE(out_split_axis, -1);
      Shape* out_shape = out_tensor->mut_shape();
      const Shape& in_shape = in_tensor->shape();
      CHECK_GT(in_shape.NumAxes(), std::max(in_split_axis, out_split_axis));
      CHECK_GT(in_shape.elem_cnt(), 0);
      CHECK_EQ(in_shape.At(out_split_axis) % parallel_num, 0);
      out_shape->Set(in_split_axis, in_shape.At(in_split_axis) * parallel_num);
      out_shape->Set(out_split_axis, in_shape.At(out_split_axis) / parallel_num);
      CHECK_EQ(out_shape->elem_cnt(), in_shape.elem_cnt());
      return Maybe<void>::Ok();
    })
    .SetInferSbpSignatureFn([](user_op::InferSbpSignatureFnContext* ctx) -> Maybe<void> {
      // S2S
      auto* bn2sbp = ctx->mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel();
      const SbpParallel& in_sbp_hint = ctx->SbpParallelHint4InputArgNameAndIndex("in", 0);
      CHECK(in_sbp_hint.has_split_parallel());
      const int64_t in_split_axis = ctx->Attr<int64_t>("in_split_axis");
      const int64_t out_split_axis = ctx->Attr<int64_t>("out_split_axis");
      CHECK_EQ(in_sbp_hint.split_parallel().axis(), in_split_axis);
      const std::string& ibn = GenRepeatedBn("in", 0);
      const std::string& obn = GenRepeatedBn("out", 0);
      (*bn2sbp)[ibn].mutable_split_parallel()->set_axis(in_split_axis);
      (*bn2sbp)[obn].mutable_split_parallel()->set_axis(out_split_axis);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
