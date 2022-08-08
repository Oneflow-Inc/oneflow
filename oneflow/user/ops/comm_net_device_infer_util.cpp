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

Maybe<bool> SyncLaunched(user_op::DeviceAndStreamInferContext* ctx) { return false; }

Maybe<bool> IsAsyncLaunched(user_op::DeviceAndStreamInferContext* ctx) {
  return ctx->Attr<bool>("async_launch");
}

namespace {

Maybe<Symbol<Stream>> RawGetNcclDevice(bool is_async_launced) {
  StreamType stream_type =
      (is_async_launced ? StreamType::kAsyncedLaunchedCommNet : StreamType::kSyncedLaunchedCommNet);
  return Stream::New(JUST(Device::New("cuda")), stream_type);
}

Maybe<Symbol<Stream>> RawGetCpuTransportDevice() {
  return Stream::New(JUST(Device::New("cpu")), StreamType::kSyncedLaunchedCommNet);
}

}  // namespace

decltype(GetNcclDevice) GetNcclDevice = DECORATE(&RawGetNcclDevice, ThreadLocal);
decltype(GetCpuTransportDevice) GetCpuTransportDevice =
    DECORATE(&RawGetCpuTransportDevice, ThreadLocal);

Maybe<Symbol<Device>> DefaultGetOutputDeivce(user_op::DeviceAndStreamInferContext* ctx) {
  CHECK_GT_OR_RETURN(ctx->inputs().size(), 0);
  return ctx->InputTensorDevice4ArgNameAndIndex("in", 0);
}

}  // namespace oneflow
