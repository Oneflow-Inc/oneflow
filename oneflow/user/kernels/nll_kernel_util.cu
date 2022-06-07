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
                           T* out, T* total_weight) {
  const T zero = GetZeroVal<T>();
  const T one = GetOneVal<T>();
  CUDA_1D_KERNEL_LOOP(i, num_samples) {
    K label = target[i];
    if (label != ignore_index) {
      label -= class_start;
      if (label >= 0 && label < num_classes) {
        const T w = weight ? weight[label] : one;
        out[i] = -(input[i * num_classes + label] * w);
        if (total_weight) { cuda::atomic::Add(total_weight, w); }
        continue;
      }
    }
    out[i] = zero;
  }
}

template<typename T, typename K>
__global__ void NLLBackward(const int32_t num_samples, const K num_classes, const K class_start,
                            const K ignore_index, const T* out_grad, const K* target,
                            const T* weight, T* in_grad) {
  const T one = GetOneVal<T>();
  CUDA_1D_KERNEL_LOOP(i, num_samples) {
    K label = target[i];
    if (label == ignore_index) { continue; }
    label -= class_start;
    if (label >= 0 && label < num_classes) {
      const T w = weight ? -weight[label] : -one;
      in_grad[i * num_classes + label] = out_grad[i] * w;
    }
  }
}

}  // namespace

template<typename T, typename K>
struct NLLKernelUtil<DeviceType::kCUDA, T, K> {
  static void Forward(ep::Stream* stream, const int32_t num_samples, const K num_classes,
                      const K class_start, const K ignore_index, const T* input, const K* target,
                      const T* weight, T* out, T* total_weight) {
    NLLForward<<<BlocksNum4ThreadsNum(num_samples), kCudaThreadsNumPerBlock, 0,
                 stream->As<ep::CudaStream>()->cuda_stream()>>>(num_samples, num_classes,
                                                                class_start, ignore_index, input,
                                                                target, weight, out, total_weight);
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
