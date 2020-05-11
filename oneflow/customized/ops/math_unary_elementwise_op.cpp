#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/ops/math_unary_elementwise_seq.h"

namespace oneflow {

#define REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD(math_unary_elementwise_type, func_prefix) \
  REGISTER_USER_OP(math_unary_elementwise_type)                                               \
      .Input("x")                                                                             \
      .Output("y")                                                                            \
      .SetTensorDescInferFn(user_op::TensorDescInferFnUtil::Unchanged)                        \
      .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)                                   \
      .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);               \
                                                                                              \
  REGISTER_USER_OP((std::string("") + math_unary_elementwise_type + "_grad"))                 \
      .Input("x")                                                                             \
      .Input("dy")                                                                            \
      .Output("dx")                                                                           \
      .SetTensorDescInferFn(user_op::TensorDescInferFnUtil::Unchanged)                        \
      .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)                                   \
      .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);               \
                                                                                              \
  REGISTER_USER_OP_GRAD(math_unary_elementwise_type)                                          \
      .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {  \
        if (op.NeedGenGradTensor4OpInput("x", 0)) {                                           \
          user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");                  \
          user_op::UserOpConfWrapper unary_grad_op =                                          \
              builder.Op(std::string("") + math_unary_elementwise_type + "_grad")             \
                  .Input("x", op.input("x", 0))                                               \
                  .Input("dy", op.GetGradTensorWithOpOutput("y", 0))                          \
                  .Output("dx")                                                               \
                  .Build();                                                                   \
          op.BindGradTensorWithOpInput(unary_grad_op.output("dx", 0), "x", 0);                \
          AddOp(unary_grad_op);                                                               \
        }                                                                                     \
      });

OF_PP_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_OP_AND_GRAD, MATH_UNARY_ELEMENTWISE_FUNC_SEQ)

}  // namespace oneflow
