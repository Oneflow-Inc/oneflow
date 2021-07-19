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

    namespace {

        Maybe<void> InferTensorDesc(user_op::InferContext *ctx) {
            const auto dim = ctx->Attr<int64_t>("axis");
            const auto sizes_list = ctx->Attr<std::vector<int64_t>>("sizes");
            const user_op::TensorDesc &in_desc = ctx->InputTensorDesc("in", 0);
            const int64_t in_num_axes = ctx->InputTensorDesc("in", 0).shape().NumAxes();
            CHECK_GE_OR_RETURN(dim, 0);
            CHECK_LT_OR_RETURN(dim, in_num_axes);

            FOR_RANGE(int64_t, i, 0, sizes_list.size())
            {
                user_op::TensorDesc *out_i_desc = ctx->OutputTensorDesc("out", i);
                DimVector out_i_dim_vec = in_desc.shape().dim_vec();
                out_i_dim_vec[dim] = sizes_list[i];
                *out_i_desc->mut_shape() = Shape(out_i_dim_vec);
            }
            return Maybe<void>::Ok();
        }

        Maybe<void> InferDataType(user_op::InferContext *ctx) {
            const user_op::TensorDesc &in_desc = ctx->InputTensorDesc("in", 0);
            const auto sizes_list = ctx->Attr<std::vector<int64_t>>("sizes");
            FOR_RANGE(int64_t, i, 0, sizes_list.size())
            {
                user_op::TensorDesc *out_i_desc = ctx->OutputTensorDesc("out", i);
                *out_i_desc->mut_data_type() = in_desc.data_type();
            }
            return Maybe<void>::Ok();
        }


        Maybe<void> GetSbpSignature(user_op::SbpContext *ctx) {
            const auto dim = ctx->Attr<int64_t>("axis");
            const int64_t in_num_axes =
                    ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
            FOR_RANGE(int64_t, i, 0, in_num_axes)
            {
                if (i == dim) { continue; }
                ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
            }
            ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
            return Maybe<void>::Ok();
        }

    }

    REGISTER_USER_OP("split_sizes")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("axis")
    .Attr<std::vector<int64_t>>("sizes")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetGetSbpFn(GetSbpSignature)
    .SetDataTypeInferFn(InferDataType);


}  // namespace oneflow
