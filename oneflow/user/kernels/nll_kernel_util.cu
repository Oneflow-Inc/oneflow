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
#include "oneflow/core/cuda/atomic.cuh"

namespace oneflow {

namespace {

template<typename T, typename K>
__global__ void NLLForward(const int32_t num_samples, const K num_classes, const K class_start,
                           const K ignore_index, const T* input, const K* target, const T* weight,
                           T* out, T* out_weight) {
  const T zero = GetZeroVal<T>();
  const T one = GetOneVal<T>();
  CUDA_1D_KERNEL_LOOP(i, num_samples) {
    K label = target[i];
    T w = zero;
    T y = zero;
    if (label != ignore_index) {
      label -= class_start;
      if (label >= 0 && label < num_classes) {
        w = weight ? weight[label] : one;
        y = -(input[i * num_classes + label] * w);
      }
    }
    out[i] = y;
    out_weight[i] = w;
  }
}

template<typename T, typename K>
__global__ void NLLBackward(const int32_t num_samples, const K num_classes, const K class_start,
                            const K ignore_index, const T* out_grad, const K* target,
                            const T* weight, T* in_grad) {
  const T one = GetOneVal<T>();
  const T zero = GetZeroVal<T>();
  CUDA_1D_KERNEL_LOOP_T(K, i, num_samples * num_classes) {
    const K n = i / num_classes;
    const K idx = i - n * num_classes;
    const K label = target[n];
    if (label != ignore_index && idx == label - class_start) {
      in_grad[i] = out_grad[n] * (weight ? -weight[idx] : -one);
    } else {
      in_grad[i] = zero;
    }
  }
}

}  // namespace

template<typename T, typename K>
struct NLLKernelUtil<DeviceType::kCUDA, T, K> {
  static void Forward(ep::Stream* stream, const int32_t num_samples, const K num_classes,
                      const K class_start, const K ignore_index, const T* input, const K* target,
                      const T* weight, T* out, T* out_weight) {
    NLLForward<<<BlocksNum4ThreadsNum(num_samples), kCudaThreadsNumPerBlock, 0,
                 stream->As<ep::CudaStream>()->cuda_stream()>>>(num_samples, num_classes,
                                                                class_start, ignore_index, input,
                                                                target, weight, out, out_weight);
  }

  static void Backward(ep::Stream* stream, const int32_t num_samples, const K num_classes,
                       const K class_start, const K ignore_index, const T* out_grad,
                       const K* target, const T* weight, T* in_grad) {
    NLLBackward<<<BlocksNum4ThreadsNum(num_samples), kCudaThreadsNumPerBlock, 0,
                  stream->As<ep::CudaStream>()->cuda_stream()>>>(
        num_samples, num_classes, class_start, ignore_index, out_grad, target, weight, in_grad);
  }
};

template struct NLLKernelUtil<DeviceType::kCUDA, float, int32_t>;
template struct NLLKernelUtil<DeviceType::kCUDA, float, int64_t>;
template struct NLLKernelUtil<DeviceType::kCUDA, double, int32_t>;
template struct NLLKernelUtil<DeviceType::kCUDA, double, int64_t>;
template struct NLLKernelUtil<DeviceType::kCUDA, half, int32_t>;
template struct NLLKernelUtil<DeviceType::kCUDA, half, int64_t>;

}  // namespace oneflow
