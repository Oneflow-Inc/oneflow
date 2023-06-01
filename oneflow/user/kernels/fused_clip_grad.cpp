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
#include "oneflow/user/kernels/fused_clip_grad_util.h"

namespace oneflow {

#define REGISTER_FUSED_CLIP_GRAD_KERNEL(device, dtype)                                         \
  REGISTER_USER_KERNEL("fused_clip_grad")                                                      \
      .SetCreateFn<FusedClipGradKernel<device, dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                    \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))       \
      .SetInferTmpSizeFn(InferFusedClipGradTempStorageSize);

REGISTER_FUSED_CLIP_GRAD_KERNEL(DeviceType::kCUDA, float);
REGISTER_FUSED_CLIP_GRAD_KERNEL(DeviceType::kCUDA, double);

}  // namespace oneflow