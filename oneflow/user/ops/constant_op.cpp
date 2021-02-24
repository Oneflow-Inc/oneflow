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
REGISTER_USER_OP("constant")
    .Output("out")
    .SetOutputBufferNum(1)
    .Attr<double>("floating_value")
    .Attr<int64_t>("integer_value")
    .Attr<bool>("is_floating_value")
    .Attr<DataType>("dtype")
    .Attr<Shape>("shape")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      const Shape& shape = ctx->Attr<Shape>("shape");
      auto dtype = ctx->Attr<DataType>("dtype");
      DimVector dim_vec;
      if (shape.NumAxes() > 0) {
        dim_vec.insert(dim_vec.end(), shape.dim_vec().cbegin(), shape.dim_vec().cend());
      }
      if (dim_vec.empty()) { dim_vec.push_back(1); }
      *ctx->Dtype4ArgNameAndIndex("out", 0) = dtype;
      *out_shape = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
