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
#include "oneflow/user/ops/math_unary_elementwise_seq.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

#define MATH_ELEMENTWISE_DEFAULT_SET_FUNC(op_type)                                       \
  /* static */ Maybe<void> op_type::InferLogicalTensorDesc(user_op::InferContext* ctx) { \
    return user_op::TensorDescInferFnUtil::Unchanged(ctx);                               \
  }                                                                                      \
  /*static*/ Maybe<void> op_type::InferPhysicalTensorDesc(user_op::InferContext* ctx) {  \
    return InferLogicalTensorDesc(ctx);                                                  \
  }                                                                                      \
  /* static */ Maybe<void> op_type::GetSbp(user_op::SbpContext* ctx) {                   \
    return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);                                 \
  }                                                                                      \
  /* static */ Maybe<void> op_type::InferDataType(user_op::InferContext* ctx) {          \
    return user_op::TensorDescInferFnUtil::UnchangedDataType(ctx);                       \
  }

// #define REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD(math_unary_elementwise_type, func_prefix)  \
//   MATH_ELEMENTWISE_DEFAULT_SET_FUNC(func_prefix##Op)                                           \
//   MATH_ELEMENTWISE_DEFAULT_SET_FUNC(func_prefix##GradOp)                                       \
//   REGISTER_USER_OP_GRAD(math_unary_elementwise_type)                                           \
//       .SetGenBackwardOpConfFn(                                                                 \
//           [](const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) -> Maybe<void> { \
//             if (op.NeedGenGradTensor4OpInput("x", 0)) {                                        \
//               user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");               \
//               user_op::UserOpConfWrapper unary_grad_op =                                       \
//                   builder.Op(std::string("") + math_unary_elementwise_type + "_grad")          \
//                       .Input("x", op.input("x", 0))                                            \
//                       .Input("dy", op.GetGradTensorWithOpOutput("y", 0))                       \
//                       .Output("dx")                                                            \
//                       .Build();                                                                \
//               op.BindGradTensorWithOpInput(unary_grad_op.output("dx", 0), "x", 0);             \
//               AddOp(unary_grad_op);                                                            \
//             }                                                                                  \
//             return Maybe<void>::Ok();                                                          \
//           });

// OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD,
//                      MATH_UNARY_ELEMENTWISE_FUNC_SEQ_ODS)

#define REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD_WITH_DY_X(math_unary_elementwise_type,     \
                                                              func_prefix)                     \
  MATH_ELEMENTWISE_DEFAULT_SET_FUNC(func_prefix##Op)                                           \
  MATH_ELEMENTWISE_DEFAULT_SET_FUNC(func_prefix##GradOp)                                       \
  REGISTER_USER_OP_GRAD(math_unary_elementwise_type)                                           \
      .SetGenBackwardOpConfFn(                                                                 \
          [](const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) -> Maybe<void> { \
            if (op.NeedGenGradTensor4OpInput("x", 0)) {                                        \
              user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");               \
              user_op::UserOpConfWrapper unary_grad_op =                                       \
                  builder.Op(std::string("") + math_unary_elementwise_type + "_grad")          \
                      .Input("x", op.input("x", 0))                                            \
                      .Input("dy", op.GetGradTensorWithOpOutput("y", 0))                       \
                      .Output("dx")                                                            \
                      .Build();                                                                \
              op.BindGradTensorWithOpInput(unary_grad_op.output("dx", 0), "x", 0);             \
              AddOp(unary_grad_op);                                                            \
            }                                                                                  \
            return Maybe<void>::Ok();                                                          \
          });

#define REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD_WITH_DY_Y(math_unary_elementwise_type,     \
                                                              func_prefix)                     \
  MATH_ELEMENTWISE_DEFAULT_SET_FUNC(func_prefix##Op)                                           \
  MATH_ELEMENTWISE_DEFAULT_SET_FUNC(func_prefix##GradOp)                                       \
  REGISTER_USER_OP_GRAD(math_unary_elementwise_type)                                           \
      .SetGenBackwardOpConfFn(                                                                 \
          [](const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) -> Maybe<void> { \
            if (op.NeedGenGradTensor4OpInput("x", 0)) {                                        \
              user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");               \
              user_op::UserOpConfWrapper unary_grad_op =                                       \
                  builder.Op(std::string("") + math_unary_elementwise_type + "_grad")          \
                      .Input("y", op.output("y", 0))                                           \
                      .Input("dy", op.GetGradTensorWithOpOutput("y", 0))                       \
                      .Output("dx")                                                            \
                      .Build();                                                                \
              op.BindGradTensorWithOpInput(unary_grad_op.output("dx", 0), "x", 0);             \
              AddOp(unary_grad_op);                                                            \
            }                                                                                  \
            return Maybe<void>::Ok();                                                          \
          });

#define REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD_WITH_FILL(math_unary_elementwise_type,     \
                                                              func_prefix)                     \
  MATH_ELEMENTWISE_DEFAULT_SET_FUNC(func_prefix##Op)                                           \
  REGISTER_USER_OP_GRAD(math_unary_elementwise_type)                                           \
      .SetGenBackwardOpConfFn(                                                                 \
          [](const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) -> Maybe<void> { \
            if (op.NeedGenGradTensor4OpInput("x", 0)) {                                        \
              user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");               \
              user_op::UserOpConfWrapper fill_op = builder.Op("fill_")                         \
                                                       .Input("in", op.input("x", 0))          \
                                                       .Output("out")                          \
                                                       .Attr<double>("floating_value", 0.)     \
                                                       .Attr<bool>("is_floating_value", true)  \
                                                       .Build();                               \
              op.BindGradTensorWithOpInput(fill_op.output("out", 0), "x", 0);                  \
              AddOp(fill_op);                                                                  \
            }                                                                                  \
            return Maybe<void>::Ok();                                                          \
          });

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD_WITH_DY_X,
                     MATH_UNARY_ELEMENTWISE_FUNC_BWD_WITH_DY_X_SEQ)

// OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD_WITH_DY_Y,
//                      MATH_UNARY_ELEMENTWISE_FUNC_BWD_WITH_DY_Y_ODS_SEQ)

MATH_ELEMENTWISE_DEFAULT_SET_FUNC(SigmoidV2Op)
MATH_ELEMENTWISE_DEFAULT_SET_FUNC(SigmoidV2GradOp)

// REGISTER_USER_OP_GRAD("sigmoid_v2")
//     .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
//                                const user_op::AddOpFn& AddOp) -> Maybe<void> {
//       if (op.NeedGenGradTensor4OpInput("x", 0)) {
//         user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
//         user_op::UserOpConfWrapper grad_op = builder.Op("sigmoid_v2_grad")
//                                                  .Input("x", op.input("x", 0))
//                                                  .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
//                                                  .Output("dx")
//                                                  .Build();
//         op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
//         AddOp(grad_op);
//       }
//       return Maybe<void>::Ok();
//     });

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD_WITH_FILL,
                     MATH_UNARY_ELEMENTWISE_FUNC_BWD_WITH_FILL_SEQ)

// test abs 
// MATH_ELEMENTWISE_DEFAULT_SET_FUNC(AbsOp)
// REGISTER_USER_OP_GRAD("abs")
//     .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
//                                const user_op::AddOpFn& AddOp) -> Maybe<void> {
//       if (op.NeedGenGradTensor4OpInput("x", 0)) {
//         user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
//         user_op::UserOpConfWrapper grad_op = builder.Op("abs_grad")
//                                                  .Input("x", op.input("x", 0))
//                                                  .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
//                                                  .Output("dx")
//                                                  .Build();
//         op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
//         AddOp(grad_op);
//       }
//       return Maybe<void>::Ok();
//     });


// For negative
MATH_ELEMENTWISE_DEFAULT_SET_FUNC(NegativeOp)
REGISTER_USER_OP_GRAD("negative")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("negative")
                                                 .Input("x", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Output("y")
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("y", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
