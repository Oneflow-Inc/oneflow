#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("invert_permutation")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn(user_op::TensorDescInferFnUtil::Unchanged)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);

}  // namespace oneflow
