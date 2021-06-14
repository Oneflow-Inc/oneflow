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

REGISTER_USER_OP("summa_matmul")
    .Input("a")
    .Input("b")
    .Output("out")
    .Attr<bool>("transpose_a", false)
    .Attr<bool>("transpose_b", false)
    .Attr<double>("alpha", 1.0)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* a = ctx->TensorDesc4ArgNameAndIndex("a", 0);
      const user_op::TensorDesc* b = ctx->TensorDesc4ArgNameAndIndex("b", 0);
      CHECK_EQ_OR_RETURN(a->shape().NumAxes(), 2);
      CHECK_EQ_OR_RETURN(b->shape().NumAxes(), 2);
      bool transpose_a = ctx->Attr<bool>("transpose_a");
      bool transpose_b = ctx->Attr<bool>("transpose_b");
      if (transpose_a && transpose_b) { UNIMPLEMENTED_THEN_RETURN(); }
      int64_t m, n, k;
      if (!transpose_a) {
        m = a->shape().At(0);
        k = a->shape().At(1);
      } else {
        m = a->shape().At(1);
        k = a->shape().At(0);
      }
      if (!transpose_b) {
        CHECK_EQ_OR_RETURN(k, b->shape().At(0));
        n = b->shape().At(1);
      } else {
        CHECK_EQ_OR_RETURN(k, b->shape().At(1));
        n = b->shape().At(0);
      }
      *ctx->Shape4ArgNameAndIndex("out", 0) = Shape({m, n});
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("a", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("a", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          ParallelDistribution parallel_distribution;
          parallel_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
          parallel_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
          *ctx->ParallelDistribution4ArgNameAndIndex("a", 0) = parallel_distribution;
          *ctx->ParallelDistribution4ArgNameAndIndex("b", 0) = parallel_distribution;
          *ctx->ParallelDistribution4ArgNameAndIndex("out", 0) = parallel_distribution;
          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("summa_matmul_placeholder")
    .Input("a")
    .Input("b")
    .Output("out")
    .Attr<double>("alpha", 1.0)
    .Attr<bool>("transpose_a", false)
    .Attr<bool>("transpose_b", false)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* a = ctx->TensorDesc4ArgNameAndIndex("a", 0);
      const user_op::TensorDesc* b = ctx->TensorDesc4ArgNameAndIndex("b", 0);
      CHECK_EQ_OR_RETURN(a->shape().NumAxes(), 2);
      CHECK_EQ_OR_RETURN(b->shape().NumAxes(), 2);
      bool transpose_a = ctx->Attr<bool>("transpose_a");
      bool transpose_b = ctx->Attr<bool>("transpose_b");
      if (transpose_a && transpose_b) { UNIMPLEMENTED_THEN_RETURN(); }
      int64_t m, n, k;
      if (!transpose_a) {
        m = a->shape().At(0);
        k = a->shape().At(1);
      } else {
        m = a->shape().At(1);
        k = a->shape().At(0);
      }
      if (!transpose_b) {
        CHECK_EQ_OR_RETURN(k, b->shape().At(0));
        n = b->shape().At(1);
      } else {
        CHECK_EQ_OR_RETURN(k, b->shape().At(1));
        n = b->shape().At(0);
      }
      *ctx->Shape4ArgNameAndIndex("out", 0) = Shape({m, n});
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("a", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("a", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          ParallelDistribution parallel_distribution;
          parallel_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
          parallel_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
          *ctx->ParallelDistribution4ArgNameAndIndex("a", 0) = parallel_distribution;
          *ctx->ParallelDistribution4ArgNameAndIndex("b", 0) = parallel_distribution;
          *ctx->ParallelDistribution4ArgNameAndIndex("out", 0) = parallel_distribution;
          return Maybe<void>::Ok();
        });

REGISTER_USER_OP_GRAD("summa_matmul_placeholder")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> void {
      double alpha = ctx->FwOp().attr<double>("alpha");
      bool transpose_a = ctx->FwOp().attr<bool>("transpose_a");
      bool transpose_b = ctx->FwOp().attr<bool>("transpose_b");
      std::string a_grad_op_name = ctx->FwOp().op_name() + "_a_grad";
      if (transpose_a && transpose_b) { UNIMPLEMENTED(); }
      if (transpose_a) {
        ctx->DefineOp(a_grad_op_name,
                      [&](user_op::BackwardOpBuilder& builder) -> user_op::UserOpConfWrapper {
                        return builder.OpTypeName("summa_matmul_placeholder")
                            .InputBind("a", ctx->FwOp().input("b", 0))
                            .InputBind("b", ctx->FwOp().output_grad("out", 0))
                            .Attr<double>("alpha", alpha)
                            .Attr<bool>("transpose_a", false)
                            .Attr<bool>("transpose_b", true)
                            .Output("out")
                            .Build();
                      });
      } else {
        ctx->DefineOp(a_grad_op_name,
                      [&](user_op::BackwardOpBuilder& builder) -> user_op::UserOpConfWrapper {
                        return builder.OpTypeName("summa_matmul_placeholder")
                            .InputBind("a", ctx->FwOp().output_grad("out", 0))
                            .InputBind("b", ctx->FwOp().input("b", 0))
                            .Attr<double>("alpha", alpha)
                            .Attr<bool>("transpose_a", false)
                            .Attr<bool>("transpose_b", true)
                            .Output("out")
                            .Build();
                      });
      }
      ctx->FwOp().InputGradBind(user_op::OpArg("a", 0), [&]() -> const std::string& {
        return ctx->GetOp(a_grad_op_name).output("out", 0);
      });
      std::string b_grad_op_name = ctx->FwOp().op_name() + "_b_grad";
      if (transpose_b) {
        ctx->DefineOp(b_grad_op_name,
                      [&](user_op::BackwardOpBuilder& builder) -> user_op::UserOpConfWrapper {
                        return builder.OpTypeName("summa_matmul_placeholder")
                            .InputBind("a", ctx->FwOp().output_grad("out", 0))
                            .InputBind("b", ctx->FwOp().input("a", 0))
                            .Attr<double>("alpha", alpha)
                            .Attr<bool>("transpose_a", true)
                            .Attr<bool>("transpose_b", false)
                            .Output("out")
                            .Build();
                      });
      } else {
        ctx->DefineOp(b_grad_op_name,
                      [&](user_op::BackwardOpBuilder& builder) -> user_op::UserOpConfWrapper {
                        return builder.OpTypeName("summa_matmul_placeholder")
                            .InputBind("a", ctx->FwOp().input("a", 0))
                            .InputBind("b", ctx->FwOp().output_grad("out", 0))
                            .Attr<double>("alpha", alpha)
                            .Attr<bool>("transpose_a", false)
                            .Attr<bool>("transpose_b", false)
                            .Output("out")
                            .Build();
                      });
      }
      ctx->FwOp().InputGradBind(user_op::OpArg("b", 0), [&]() -> const std::string& {
        return ctx->GetOp(b_grad_op_name).output("out", 0);
      });
    });

}  // namespace oneflow
