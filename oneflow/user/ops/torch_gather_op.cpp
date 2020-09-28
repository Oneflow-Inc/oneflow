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

namespace user_op{
REGISTER_USER_OP("torch_gather")
    .Input("input")
    .Input("index")
    .Output("out")
    .Attr("dim", UserOpAttrType::kAtInt64)
    .Attr("sparse_grad", UserOpAttrType::kAtBool)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("input", 0);
      CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);

      const TensorDesc* index = ctx->TensorDesc4ArgNameAndIndex("index", 0);
      CHECK_GT_OR_RETURN(index->shape().NumAxes(), 0);
      CHECK_OR_RETURN(IsIndexDataType(index->data_type()));

      const int64_t dim = ctx->Attr<int64_t>("dim");
      CHECK_GE_OR_RETURN(dim, 0);

      // check in and index tensor, only axis "dim" differs
      // ...

      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out->mut_shape() = index->shape();
      *out->mut_data_type() = in->data_type();

      return Maybe<void>::Ok();
    });
} // namespace user_op

}  // namespace oneflow
