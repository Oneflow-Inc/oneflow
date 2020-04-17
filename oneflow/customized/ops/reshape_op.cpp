#include <vector>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_registration.h"

namespace oneflow {

REGISTER_USER_OP("reshape")
    .Input("input")
    .Output("output")
    .Attr("shape", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn([](user_op::InferContext *ctx) -> Maybe<void> {
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("input", 0);
        Shape* out_shape = ctx->Shape4ArgNameAndIndex("output", 0);
        const auto& shape = ctx->GetAttr<std::vector<int32_t>>("shape");
        CHECK_GE_OR_RETURN(shape.size(), 1);
        DimVector dim_vec = {shape.begin(), shape.end()};
        FOR_RANGE(int32_t, i, 0, dim_vec.size()) {
            CHECK_GT_OR_RETURN(dim_vec.at(i), 0);
        }
        const auto& sbp_parallel_it = ctx->SbpParallel4ArgNameAndIndex("output", 0);
        const auto& parallel_ctx_it = ctx->parallel_ctx();

        
        
    })



}