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
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<Symbol<Stream>> MakeCopyStream(const Symbol<Device>& in_device,
                                     const Symbol<Device>& out_device, const bool pin_memory) {
  if (in_device->type() != "cpu" && out_device->type() == "cpu") {
    return Stream::New(in_device, StreamRole::kDevice2Host);
  } else if (in_device->type() == "cpu" && out_device->type() != "cpu") {
    const auto device = JUST(Device::New(out_device->type(), out_device->device_id()));
    return Stream::New(device, StreamRole::kHost2Device);
  } else if (in_device->type() == "cpu" && out_device->type() == "cpu" && pin_memory) {
    // TODO:(zhaoluyang) Parsing pin-memory-device from python
    auto pin_device = JUST(Device::New("cuda"));
    return Stream::New(pin_device, StreamRole::kPinnedCompute);
  } else {
    CHECK_EQ_OR_RETURN(in_device->type(), out_device->type());
    return Stream::New(out_device, StreamRole::kCompute);
  }
}

}  // namespace

/* static */ Maybe<void> CopyOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->MutOutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->MutOutputStride("out", 0) = ctx->InputStride("in", 0);
  *ctx->MutOutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CopyOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CopyOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& inputs = ctx->inputs();
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  const auto& input =
      ctx->LogicalTensorDesc4InputArgNameAndIndex(inputs[0].first, inputs[0].second);
  for (int64_t axis = 0; axis < input.shape().NumAxes(); ++axis) {
    ctx->NewBuilder().Split(inputs, axis).Split(ctx->outputs(), axis).Build();
  }
  ctx->NewBuilder().PartialSum(inputs).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CopyOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> CopyOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  Symbol<Device> out_device =
      JUST(Device::New(ctx->Attr<std::string>("device_type"), ctx->Attr<int64_t>("device_id")));
  *ctx->OutputTensorDevice4ArgNameAndIndex("out", 0) = out_device;
  const Symbol<Device>& in_device = ctx->InputTensorDevice4ArgNameAndIndex("in", 0);
  const bool pin_memory = ctx->Attr<bool>("pin_memory");
  return MakeCopyStream(in_device, out_device, pin_memory);
}

}  // namespace oneflow
