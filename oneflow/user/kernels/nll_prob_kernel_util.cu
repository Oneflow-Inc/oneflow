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

namespace {

template<typename T, bool HasLabelSmoothing>
__global__ void NLLProbForward(const int32_t num_samples, const int64_t num_classes, const T* input,
                               const T* probs, const T* weight, const T one_minus_label_smoothing,
                               const T label_smoothing_rest_factor, T* out) {
  const T one = GetOneVal<T>();
  CUDA_1D_KERNEL_LOOP(i, num_samples * num_classes) {
    T prob = HasLabelSmoothing ? probs[i] * one_minus_label_smoothing + label_smoothing_rest_factor
                               : probs[i];
    T w = weight ? weight[i % num_classes] : one;
    T y = -input[i] * w * prob;
    out[i] = y;
  }
}

template<typename T, bool HasLabelSmoothing>
__global__ void NLLProbBackward(const int32_t num_samples, const int64_t num_classes,
                                const T* out_grad, const T* probs, const T* weight,
                                const T one_minus_label_smoothing,
                                const T label_smoothing_rest_factor, T* in_grad) {
  const T one = GetOneVal<T>();
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, num_samples * num_classes) {
    T prob = HasLabelSmoothing ? probs[i] * one_minus_label_smoothing + label_smoothing_rest_factor
                               : probs[i];
    T w = weight ? weight[i % num_classes] : one;
    in_grad[i] = -w * prob * out_grad[i];
  }
}

}  // namespace

template<typename T, bool HasLabelSmoothing>
struct NLLProbKernelUtil<DeviceType::kCUDA, T, HasLabelSmoothing> {
  static void Forward(ep::Stream* stream, const int32_t num_samples, const int64_t num_classes,
                      const T* input, const T* probs, const T* weight,
                      const T one_minus_label_smoothing, const T label_smoothing_rest_factor,
                      T* out) {
    NLLProbForward<T, HasLabelSmoothing>
        <<<BlocksNum4ThreadsNum(num_samples), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(num_samples, num_classes, input, probs,
                                                          weight, one_minus_label_smoothing,
                                                          label_smoothing_rest_factor, out);
  }

  static void Backward(ep::Stream* stream, const int32_t num_samples, const int64_t num_classes,
                       const T* out_grad, const T* probs, const T* weight,
                       const T one_minus_label_smoothing, const T label_smoothing_rest_factor,
                       T* in_grad) {
    NLLProbBackward<T, HasLabelSmoothing>
        <<<BlocksNum4ThreadsNum(num_samples), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(num_samples, num_classes, out_grad, probs,
                                                          weight, one_minus_label_smoothing,
                                                          label_smoothing_rest_factor, in_grad);
  }
};

template struct NLLProbKernelUtil<DeviceType::kCUDA, float, false>;
template struct NLLProbKernelUtil<DeviceType::kCUDA, float, true>;
template struct NLLProbKernelUtil<DeviceType::kCUDA, double, false>;
template struct NLLProbKernelUtil<DeviceType::kCUDA, double, true>;
template struct NLLProbKernelUtil<DeviceType::kCUDA, half, false>;
template struct NLLProbKernelUtil<DeviceType::kCUDA, half, true>;

}  // namespace oneflow
