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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> EagerSToPOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, Shape(ctx->Attr<Shape>("shape").dim_vec()));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerSToPOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EagerSToPOp::GetSbp(user_op::SbpContext* ctx) {
  return Error::TypeError() << "eager_b_to_s op doesn't support global tensor!";
}

/* static */ Maybe<void> EagerSToPOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  return Error::TypeError() << "eager_b_to_s op doesn't support global tensor!";
}

/* static */ Maybe<void> EagerSToPOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> EagerSToPOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

}  // namespace oneflow
