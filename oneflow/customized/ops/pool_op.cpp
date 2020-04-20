#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

REGISTER_USER_OP("avg_pool_1d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(PoolOpUtil::MakeFwTensorDescInferFn(1))
    .SetBatchAxisInferFn(PoolOpUtil::MakeFwBatchAxisInferFn())
    .SetGetSbpFn(PoolOpUtil::MakeFwGetSbpFn());

REGISTER_USER_OP("avg_pool_1d_grad")
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

REGISTER_USER_OP_GRAD("avg_pool_1d")
    .SetGenBackwardOpConfFn(PoolOpUtil::MakeGenBackwardOpConfFn("avg", 1));

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

REGISTER_USER_OP("avg_pool_3d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(PoolOpUtil::MakeFwTensorDescInferFn(3))
    .SetBatchAxisInferFn(PoolOpUtil::MakeFwBatchAxisInferFn())
    .SetGetSbpFn(PoolOpUtil::MakeFwGetSbpFn());

REGISTER_USER_OP("avg_pool_3d_grad")
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

REGISTER_USER_OP_GRAD("avg_pool_3d")
    .SetGenBackwardOpConfFn(PoolOpUtil::MakeGenBackwardOpConfFn("avg", 3));

REGISTER_USER_OP("max_pool_1d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(PoolOpUtil::MakeFwTensorDescInferFn(1))
    .SetBatchAxisInferFn(PoolOpUtil::MakeFwBatchAxisInferFn())
    .SetGetSbpFn(PoolOpUtil::MakeFwGetSbpFn());

REGISTER_USER_OP("max_pool_1d_grad")
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

REGISTER_USER_OP_GRAD("max_pool_1d")
    .SetGenBackwardOpConfFn(PoolOpUtil::MakeGenBackwardOpConfFn("max", 1));

REGISTER_USER_OP("max_pool_2d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(PoolOpUtil::MakeFwTensorDescInferFn(2))
    .SetBatchAxisInferFn(PoolOpUtil::MakeFwBatchAxisInferFn())
    .SetGetSbpFn(PoolOpUtil::MakeFwGetSbpFn());

REGISTER_USER_OP("max_pool_2d_grad")
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

REGISTER_USER_OP_GRAD("max_pool_2d")
    .SetGenBackwardOpConfFn(PoolOpUtil::MakeGenBackwardOpConfFn("max", 2));

REGISTER_USER_OP("max_pool_3d")
    .Input("x")
    .Output("y")
    .Attr("padding", UserOpAttrType::kAtString)
    .Attr("data_format", UserOpAttrType::kAtString)
    .Attr("pool_size", UserOpAttrType::kAtListInt32)
    .Attr("strides", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn(PoolOpUtil::MakeFwTensorDescInferFn(3))
    .SetBatchAxisInferFn(PoolOpUtil::MakeFwBatchAxisInferFn())
    .SetGetSbpFn(PoolOpUtil::MakeFwGetSbpFn());

REGISTER_USER_OP("max_pool_3d_grad")
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

REGISTER_USER_OP_GRAD("max_pool_3d")
    .SetGenBackwardOpConfFn(PoolOpUtil::MakeGenBackwardOpConfFn("max", 3));

}  // namespace oneflow
