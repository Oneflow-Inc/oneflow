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
#include "oneflow/user/ops/comm_net_device_infer_util.h"

namespace oneflow {

Maybe<bool> SyncLaunched(user_op::DeviceInferContext* ctx) { return false; }

Maybe<bool> IsAsyncLaunched(user_op::DeviceInferContext* ctx) {
  return ctx->Attr<bool>("async_launch");
}

namespace {

Maybe<Symbol<Stream>> RawGetNcclStream(bool is_async_launced) {
  return Stream::NewByDefaultDevice(is_async_launced ? "async_launched_nccl" : "sync_launched_nccl");
}

Maybe<Symbol<Stream>> RawGetCpuTransportDevice() { return Stream::NewByDefaultDevice("comm_net"); }

}  // namespace

decltype(GetNcclStream) GetNcclStream = DECORATE(&RawGetNcclStream, ThreadLocal);
decltype(GetCpuTransportStream) GetCpuTransportStream =
    DECORATE(&RawGetCpuTransportStream, ThreadLocal);

Maybe<Symbol<Device>> DefaultGetOutputDeivce(user_op::DeviceInferContext* ctx) {
  CHECK_GT_OR_RETURN(ctx->inputs().size(), 0);
  return ctx->InputTensorDevice4ArgNameAndIndex("in", 0);
}

}  // namespace oneflow
