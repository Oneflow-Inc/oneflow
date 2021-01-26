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

REGISTER_USER_OP("hierarchical_parallel_cast")
    .Input("in")
    .Output("out")
    .Attr<Shape>("parallel_hierarchy")
    .Attr<std::vector<std::string>>("parallel_distribution")
    .Attr<std::string>("grad_mode")
    .Attr<Shape>("grad_parallel_hierarchy")
    .Attr<std::vector<std::string>>("grad_parallel_distribution")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *ctx->TensorDesc4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetInferParallelHierarchyFn([](user_op::InferParallelHierarchyFnContext* ctx) -> Maybe<void> {
      const Shape parallel_hierarchy = ctx->user_op_conf().attr<Shape>("parallel_hierarchy");
      CHECK_EQ_OR_RETURN(parallel_hierarchy.elem_cnt(), ctx->parallel_num());
      *ctx->mut_parallel_hierarchy() = parallel_hierarchy;
      return Maybe<void>::Ok();
    })
    .SetInferSbpSignatureFn([](user_op::InferSbpSignatureFnContext* ctx) -> Maybe<void> {
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("hierarchical_parallel_cast_like")
    .Input("in")
    .Input("like")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *ctx->TensorDesc4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetInferParallelHierarchyFn([](user_op::InferParallelHierarchyFnContext* ctx) -> Maybe<void> {
      *ctx->mut_parallel_hierarchy() = ctx->ParallelHierarchy4InputArgNameAndIndex("like", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetInferSbpSignatureFn([](user_op::InferSbpSignatureFnContext* ctx) -> Maybe<void> {
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("hierarchical_parallel_cast")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      if (ctx->FwOp().NeedGenGradTensor4OpInput("in", 0)) {
        const auto& grad_mode = ctx->FwOp().attr<std::string>("grad_mode");
        if (grad_mode == "identity") {
          ctx->FwOp().BindGradTensorWithOpInput(ctx->FwOp().GetGradTensorWithOpOutput("out", 0),
                                                "in", 0);
        } else if (grad_mode == "manual") {
          const std::string grad_op_name = "System-AutoGrad-" + ctx->FwOp().op_name();
          ctx->DefineOp(grad_op_name, [&](user_op::BackwardOpBuilder& builder) {
            return builder.OpTypeName("hierarchical_parallel_cast")
                .InputBind("in", ctx->FwOp().output_grad("out", 0))
                .Output("out")
                .Attr<Shape>("parallel_hierarchy",
                             ctx->FwOp().attr<Shape>("grad_parallel_hierarchy"))
                .Attr<std::vector<std::string>>(
                    "parallel_distribution",
                    ctx->FwOp().attr<std::vector<std::string>>("grad_parallel_distribution"))
                .Attr<Shape>("grad_parallel_hierarchy", Shape())
                .Attr<std::vector<std::string>>("grad_parallel_distribution",
                                                std::vector<std::string>())
                .Build();
          });
          ctx->FwOp().InputGradBind(user_op::OpArg("in", 0), [&]() -> const std::string& {
            return ctx->GetOp(grad_op_name).output("out", 0);
          });
        } else if (grad_mode == "restore") {
          const std::string grad_op_name = "System-AutoGrad-" + ctx->FwOp().op_name();
          ctx->DefineOp(grad_op_name, [&](user_op::BackwardOpBuilder& builder) {
            return builder.OpTypeName("hierarchical_parallel_cast_like")
                .InputBind("in", ctx->FwOp().output_grad("out", 0))
                .InputBind("like", ctx->FwOp().input("like", 0))
                .Output("out")
                .Build();
          });
          ctx->FwOp().InputGradBind(user_op::OpArg("in", 0), [&]() -> const std::string& {
            return ctx->GetOp(grad_op_name).output("out", 0);
          });
        } else {
          UNIMPLEMENTED();
        }
      }
    });

}  // namespace oneflow
