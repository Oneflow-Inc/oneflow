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
#include "oneflow/core/framework/user_op_attr.pb.h"

namespace oneflow {

namespace summary {

Maybe<void> CheckStepShape(const Shape* step) {
  CHECK_OR_RETURN(step->elem_cnt() == 1);
  return Maybe<void>::Ok();
}

REGISTER_CPU_ONLY_USER_OP("create_summary_writer")
    .Attr<std::string>("logdir")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("flush_summary_writer")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("summary_write_scalar")
    .Input("in")
    .Input("step")
    .Input("tag")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      const Shape* step_shape = ctx->Shape4ArgNameAndIndex("step", 0);
      CHECK_OR_RETURN(in_shape->elem_cnt() == 1 && step_shape->elem_cnt() == 1);
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("summary_write_histogram")
    .Input("in")
    .Input("step")
    .Input("tag")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      CheckStepShape(ctx->Shape4ArgNameAndIndex("step", 0));
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("summary_write_pb")
    .Input("in")
    .Input("step")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      CheckStepShape(ctx->Shape4ArgNameAndIndex("step", 0));
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("summary_write_image")
    .Input("in")
    .Input("step")
    .Input("tag")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      CheckStepShape(ctx->Shape4ArgNameAndIndex("step", 0));
      return Maybe<void>::Ok();
    });

}  // namespace summary

}  // namespace oneflow
