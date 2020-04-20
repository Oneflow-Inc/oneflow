#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

REGISTER_USER_OP("avg_pool_2d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(PoolOpUtil::MakeFwTensorDescInferFn(2))
    .SetBatchAxisInferFn(PoolOpUtil::MakeFwBatchAxisInferFn())
    .SetGetSbpFn(PoolOpUtil::MakeFwGetSbpFn());

REGISTER_USER_OP("avg_pool_2d_grad")
    .Input("x")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(PoolOpUtil::MakeBwTensorDescInferFn())
    .SetBatchAxisInferFn(PoolOpUtil::MakeBwBatchAxisInferFn())
    .SetGetSbpFn(PoolOpUtil::MakeBwGetSbpFn());

REGISTER_USER_OP_GRAD("avg_pool_2d")
    .SetGenBackwardOpConfFn(PoolOpUtil::MakeGenBackwardOpConfFn("avg", 2));

}  // namespace oneflow
