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
#include "oneflow/user/ops/comm_net_device_infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ Maybe<void> SendOp::GetSbp(user_op::SbpContext* ctx) { UNIMPLEMENTED_THEN_RETURN(); }
/*static*/ Maybe<void> SendOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  // Do nothing.
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SendOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return SendOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SendOp::InferDataType(user_op::InferContext* ctx) {
  // Do nothing.
  return Maybe<void>::Ok();
}
/*static*/ Maybe<Symbol<Stream>> SendOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

namespace {

Maybe<Symbol<Device>> GetRecvOutputDeivce(user_op::DeviceAndStreamInferContext* ctx) {
  const std::string& device_type = ctx->Attr<std::string>("device_type");
  const int device_id = ctx->Attr<int64_t>("device_id");
  return Device::New(device_type, device_id);
}

}  // namespace

/*static*/ Maybe<void> RecvOp::GetSbp(user_op::SbpContext* ctx) { UNIMPLEMENTED_THEN_RETURN(); }
/*static*/ Maybe<void> RecvOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->Attr<Shape>("shape"));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> RecvOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return RecvOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> RecvOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->Attr<DataType>("dtype"));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<Symbol<Stream>> RecvOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn<&GetRecvOutputDeivce>(ctx);
}

}  // namespace oneflow
