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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, const int64_t* train_step, const float* learning_rate,
                               T decay_rate, T epsilon, bool centered, T weight_decay,
                               const T* model_diff, T* model, T* mean_square, T* mean_gradient) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T model_diff_val = model_diff[i];
    mean_square[i] =
        (1 - decay_rate) * model_diff_val * model_diff_val + decay_rate * mean_square[i];
    T denom_t;
    if (centered) {
      mean_gradient[i] = (1 - decay_rate) * model_diff_val + decay_rate * mean_gradient[i];
      denom_t = mean_square[i] - mean_gradient[i] * mean_gradient[i];
    } else {
      denom_t = mean_square[i];
    }
    model[i] = model[i] - *learning_rate * model_diff_val / std::sqrt(denom_t + epsilon);
  }
}

}  // namespace

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const int64_t* train_step,
                          const float* learning_rate, T decay_rate, T epsilon, bool centered,
                          T weight_decay, const T* model_diff, T* model, T* mean_square,
                          T* mean_gradient) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, train_step, learning_rate, decay_rate, epsilon, centered, weight_decay, model_diff,
        model, mean_square, mean_gradient);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class RMSPropMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
