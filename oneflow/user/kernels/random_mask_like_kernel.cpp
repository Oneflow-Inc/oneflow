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
#include "oneflow/user/kernels/random_mask_like_kernel.h"

namespace oneflow {

namespace {
#define REGISTER_RANDOM_MASK_LIKE_KERNEL(device)   \
  REGISTER_USER_KERNEL("random_mask_like")         \
      .SetCreateFn<RandomMaskLikeKernel<device>>() \
      .SetIsMatchedHob(user_op::HobDeviceType() == device);

REGISTER_RANDOM_MASK_LIKE_KERNEL(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_RANDOM_MASK_LIKE_KERNEL(DeviceType::kCUDA)
#endif
}  // namespace

}  // namespace oneflow
