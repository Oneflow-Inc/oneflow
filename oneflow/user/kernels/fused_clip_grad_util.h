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
#ifndef ONEFLOW_USER_KERNELS_FUSED_CLIP_GRAD_UTIL_H_
#define ONEFLOW_USER_KERNELS_FUSED_CLIP_GRAD_UTIL_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

template<typename T>
struct MultiClipGradParam {
  T* data;
  size_t size;
};

enum ClipGradType : int {
  ZeroType,
  PowerType,
  OtherType,
};

template<DeviceType device_type, typename T>
struct MultiClipGrad {
  void operator()(ep::Stream* stream, std::vector<MultiClipGradParam<T>>& params, T* scale,
                  const float norm_type, const float max_norm, const ClipGradType clip_grad_type);
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_FUSED_CLIP_GRAD_UTIL_H_