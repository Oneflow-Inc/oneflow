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
#include "oneflow/user/kernels/unfold_tensor_kernel_utils.h"

namespace oneflow {

REGISTER_USER_OP("unfold_tensor")
    .Input("x")
    .Output("y")
    .Attr<int32_t>("dimension")
    .Attr<int32_t>("size")
    .Attr<int32_t>("step")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->InputTensorDesc("x", 0);
      const int32_t dimension = ctx->Attr<int32_t>("dimension");
      const int32_t size = ctx->Attr<int32_t>("size");
      const int32_t step = ctx->Attr<int32_t>("step");

      const Shape& in_shape = ctx->InputShape("x", 0);
      const int32_t in_dim = in_shape.NumAxes();
      CHECK_GE_OR_RETURN(dimension, 0);
      CHECK_LE_OR_RETURN(dimension, in_dim - 1);

      const int32_t max_size = in_dim == 0 ? 1 : in_shape.At(dimension);
      CHECK_GT_OR_RETURN(size, 0);
      CHECK_LE_OR_RETURN(size, max_size);
      CHECK_GT_OR_RETURN(step, 0);

      DimVector out_shape(in_dim + 1);
      out_shape[in_dim] = size;
      FOR_RANGE(int32_t, d, 0, in_dim) {
        int32_t in_size_at_d = in.shape().At(d);
        if (d == dimension) {
          out_shape.at(d) = (in_size_at_d - size) / step + 1;
        } else {
          out_shape.at(d) = in_size_at_d;
        }
      }
      *ctx->OutputShape("y", 0) = Shape(out_shape);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
