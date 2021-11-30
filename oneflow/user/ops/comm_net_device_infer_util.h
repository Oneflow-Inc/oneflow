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
#ifndef ONEFLOW_CORE_USER_OP_NCCL_DEVICE_INFER_UTIL_H_
#define ONEFLOW_CORE_USER_OP_NCCL_DEVICE_INFER_UTIL_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

Maybe<bool> SyncLaunched(user_op::DeviceInferContext* ctx);

Maybe<bool> IsAsyncLaunched(user_op::DeviceInferContext* ctx);

extern Maybe<Symbol<Device>> (*GetNcclDevice)(bool is_async_launced);
extern Maybe<Symbol<Device>> (*GetCpuTransportDevice)();

Maybe<Symbol<Device>> DefaultGetOutputDeivce(user_op::DeviceInferContext* ctx);

template<
    Maybe<bool> (*GetIsAsyncLaunched)(user_op::DeviceInferContext*),
    Maybe<Symbol<Device>> (*GetOutputDeivce)(user_op::DeviceInferContext*) = DefaultGetOutputDeivce>
Maybe<Symbol<Device>> DeviceInferFn(user_op::DeviceInferContext* ctx) {
  Symbol<Device> output_device = JUST(GetOutputDeivce(ctx));
  if (ctx->outputs().size() > 0) {
    *ctx->OutputTensorDevice4ArgNameAndIndex("out", 0) = output_device;
  }
  if (output_device->type() == "cuda" || output_device->type() == "gpu") {
    bool is_async_launched = JUST(GetIsAsyncLaunched(ctx));
    const auto& cuda_device = JUST(GetNcclDevice(is_async_launched));
    return cuda_device;
  } else if (output_device->type() == "cpu") {
    return JUST(GetCpuTransportDevice());
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_USER_OP_NCCL_DEVICE_INFER_UTIL_H_
