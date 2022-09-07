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
#include "oneflow/user/kernels/nll_prob_kernel_util.h"

namespace oneflow {

template<typename T>
struct NLLProbKernelUtil<DeviceType::kCPU, T> {
  static void Forward(ep::Stream* stream, const int32_t num_samples, const int64_t num_classes,
                      const T* input, const T* probs, const T* weight, const double label_smoothing,
                      T* out) {
    FOR_RANGE(int32_t, i, 0, num_samples * num_classes) {
      T prob = label_smoothing ? probs[i] * (T{1} - static_cast<T>(label_smoothing))
                                     + static_cast<T>(label_smoothing) / static_cast<T>(num_classes)
                               : probs[i];
      std::cout << "prob: " << prob << " input: " << input[i] << std::endl;
      T w = weight ? weight[i % num_classes] : T{1};
      T y = -input[i] * w * prob;
      out[i] = y;
    }
  }

  static void Backward(ep::Stream* stream, const int32_t num_samples, const int64_t num_classes,
                       const T* out_grad, const T* probs, const T* weight,
                       const double label_smoothing, T* in_grad) {
    Memset<DeviceType::kCPU>(stream, in_grad, 0,
                             RoundUp(num_samples * num_classes * sizeof(T), kBlobBodyAlignSize));
    FOR_RANGE(int32_t, i, 0, num_samples * num_classes) {
      T prob = label_smoothing ? probs[i] * (T{1} - static_cast<T>(label_smoothing))
                                     + static_cast<T>(label_smoothing) / static_cast<T>(num_classes)
                               : probs[i];
      T w = weight ? weight[i % num_classes] : T{1};
      in_grad[i] = -w * prob * out_grad[i];
    }
  }
};

template struct NLLProbKernelUtil<DeviceType::kCPU, float>;
// template struct NLLProbKernelUtil<DeviceType::kCPU, float>;
template struct NLLProbKernelUtil<DeviceType::kCPU, double>;
// template struct NLLProbKernelUtil<DeviceType::kCPU, double>;

}  // namespace oneflow
