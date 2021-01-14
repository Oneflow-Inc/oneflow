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
using namespace std;

namespace oneflow {
namespace {
Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
    std::cout << "*****************diag_op****************" << std::endl;
    const user_op::TensorDesc* input_tensor = ctx->TensorDesc4ArgNameAndIndex("input_tensor", 0);
    const int32_t dimension = ctx->Attr<int32_t>("dimension");
    const ShapeView& in_shape = input_tensor->shape();
    const int32_t in_dim = in_shape.NumAxes();
    int32_t output_dim = (in_dim == 1 ? 2 : 1);
    DimVector out_dim_vec = {output_dim, 0};

    if (in_dim == 1) {
        int32_t out_tensor_size = in_shape.At(0) + std::abs(dimension);
        out_dim_vec[0] = out_tensor_size;
        out_dim_vec[1] = out_tensor_size;
    } else {
        if (dimension >= 0) {
                out_dim_vec[0] = std::min(in_shape.At(0), in_shape.At(1) - dimension);
            } else {
                out_dim_vec[0] = std::min(in_shape.At(0) + dimension, in_shape.At(1));
                
            }
    }

    user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("diag_out", 0);
    out_desc->set_is_dynamic(false);
    *out_desc->mut_shape() = Shape(out_dim_vec);
    *out_desc->mut_data_type() = oneflow::kFloat;
    //*out_desc->mut_data_type() = input_tensor->data_type();
    return Maybe<void>::Ok();

}

Maybe<void> GetSbpSignatures4Diag(user_op::SbpContext* ctx) {
    const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("input_tensor", 0);
    int32_t axis = in_tensor.shape().NumAxes();
    FOR_RANGE(int32_t, i, 0, axis) {
        if (i == axis) { continue; }
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    }
    ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
    return Maybe<void>::Ok();
}

}

REGISTER_USER_OP("diag")
    .Input("input_tensor")
    .Output("diag_out")
    .Attr<int32_t>("dimension", 0)
    .SetTensorDescInferFn(InferTensorDesc)
    .SetGetSbpFn(GetSbpSignatures4Diag)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);

REGISTER_USER_OP_GRAD("concat").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp){
    if (op.NeedGenGradTensor4OpInput("input_tensor", 0)){
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = 
            builder.Op("diag_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("dy", 0))
                .Attr("dimension", op.attr<int32_t>("dimension"))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "input_tensor", 0);
        AddOp(grad_op);
    }
});
}  // namespace oneflow