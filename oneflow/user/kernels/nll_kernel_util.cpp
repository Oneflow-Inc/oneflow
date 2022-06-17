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
#include "oneflow/user/kernels/nll_kernel_util.h"

namespace oneflow {

template<typename T, typename K>
struct NLLKernelUtil<DeviceType::kCPU, T, K> {
  static void Forward(ep::Stream* stream, const int32_t num_samples, const K num_classes,
                      const K class_start, const K ignore_index, const T* input, const K* target,
                      const T* weight, T* out, T* out_weight) {
    FOR_RANGE(int32_t, i, 0, num_samples) {
      K label = target[i];
      T w = T{0};
      T y = T{0};
      if (label != ignore_index) {
        label -= class_start;
        if (label >= 0 && label < num_classes) {
          w = weight ? weight[label] : T{1};
          y = -(input[i * num_classes + label] * w);
        }
      }
      out[i] = y;
      out_weight[i] = w;
    }
  }

  static void Backward(ep::Stream* stream, const int32_t num_samples, const K num_classes,
                       const K class_start, const K ignore_index, const T* out_grad,
                       const K* target, const T* weight, T* in_grad) {
    Memset<DeviceType::kCPU>(stream, in_grad, 0,
                             RoundUp(num_samples * num_classes * sizeof(T), kBlobBodyAlignSize));
    FOR_RANGE(int32_t, i, 0, num_samples) {
      K label = target[i];
      if (label == ignore_index) { continue; }
      label -= class_start;
      if (label >= 0 && label < num_classes) {
        const T w = weight ? -weight[label] : T(-1);
        in_grad[i * num_classes + label] = out_grad[i] * w;
      }
    }
  }
};

template struct NLLKernelUtil<DeviceType::kCPU, float, int32_t>;
template struct NLLKernelUtil<DeviceType::kCPU, float, int64_t>;
template struct NLLKernelUtil<DeviceType::kCPU, double, int32_t>;
template struct NLLKernelUtil<DeviceType::kCPU, double, int64_t>;

}  // namespace oneflow
