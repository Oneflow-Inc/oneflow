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
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/user/ops/comm_net_device_infer_util.h"

namespace oneflow {

// Can only be called in mirrored
REGISTER_NO_GRAD_USER_OP("eager_s_to_b")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("in_split_axis", -1)
    .Attr<std::string>("in_parallel_conf")
    .Attr<std::string>("out_parallel_conf")
    .Attr<Shape>("shape")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = Shape(ctx->Attr<Shape>("shape").dim_vec());
      return Maybe<void>::Ok();
    })
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      return Error::TypeError() << "eager_s_to_b op doesn't support consistent tensor!";
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      return Error::TypeError() << "eager_s_to_b op doesn't support consistent tensor!";
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
