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

namespace {

Maybe<void> TensorDescInfer(user_op::InferContext* ctx) {
  const Shape& shape = ctx->Attr<Shape>("shape");
  const std::string& out_parallel_conf_txt = ctx->Attr<std::string>("out_parallel_conf");
  const int64_t out_split_axis = ctx->Attr<int64_t>("out_split_axis");
  Symbol<ParallelDesc> out_parallel_desc = JUST(TxtStringToPlacement(out_parallel_conf_txt));
  DimVector dim_vec{shape.dim_vec()};
  int64_t out_parallel_num = out_parallel_desc->parallel_num();
  if (out_parallel_num > 1) {
    CHECK_LE_OR_RETURN(out_split_axis, shape.NumAxes());
    CHECK_OR_RETURN(shape.At(out_split_axis) % out_parallel_num == 0);
    dim_vec[out_split_axis] = shape.At(out_split_axis) / out_parallel_num;
  }
  *ctx->OutputShape("out", 0) = Shape(dim_vec);
  return Maybe<void>::Ok();
}

}  // namespace

// Can only be called in mirrored
REGISTER_NO_GRAD_USER_OP("eager_naive_s_to_s")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("in_split_axis", -1)
    .Attr<int64_t>("out_split_axis", -1)
    .Attr<std::string>("in_parallel_conf")
    .Attr<std::string>("out_parallel_conf")
    .Attr<Shape>("shape")
    .SetTensorDescInferFn(TensorDescInfer)
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      return Error::TypeError() << "eager_naive_s_to_s op doesn't support consistent tensor!";
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      return Error::TypeError() << "eager_naive_s_to_s op doesn't support consistent tensor!";
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
