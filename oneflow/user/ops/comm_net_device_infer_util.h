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
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/common/env_var/eager.h"
#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {

extern Maybe<Symbol<Stream>> (*GetTransportDevice)(Symbol<Device>);

Maybe<Symbol<Device>> DefaultGetOutputDeivce(user_op::DeviceAndStreamInferContext* ctx);

template<Maybe<Symbol<Device>> (*GetOutputDeivce)(user_op::DeviceAndStreamInferContext*) =
             DefaultGetOutputDeivce>
Maybe<Symbol<Stream>> DeviceAndStreamInferFn(user_op::DeviceAndStreamInferContext* ctx) {
  Symbol<Device> output_device = JUST(GetOutputDeivce(ctx));
  for (const auto& pair : ctx->outputs()) {
    *ctx->OutputTensorDevice4ArgNameAndIndex(pair.first, pair.second) = output_device;
  }
  if (EagerNcclUseComputeStream() && !LazyMode::is_enabled()) {
    return GetDefaultStreamByDevice(output_device);
  }
  return GetTransportDevice(output_device);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_USER_OP_NCCL_DEVICE_INFER_UTIL_H_
