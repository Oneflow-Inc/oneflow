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
#include "oneflow/core/kernel/adam_model_update_kernel.h"

namespace oneflow {

namespace {

template<int32_t power>
struct PowUtil;

template<>
struct PowUtil<1> final {
  template<typename T>
  __device__ static T pow(const T x) {
    return x;
  }
};

template<>
struct PowUtil<2> final {
  template<typename T>
  __device__ static T pow(const T x) {
    return x * x;
  }
};

template<int32_t power, typename T>
__device__ void UpdateMomentEstimate(T beta, const T model_diff, T* moment) {
  // Update biased moment estimate
  *moment = beta * (*moment) + (1 - beta) * PowUtil<power>::pow(model_diff);
}

template<typename T>
__device__ void UpdateModel(const float learning_rate, T weight_decay, T epsilon, T* model,
                            const T m, const T v) {
  T model_val = *model;
  T model_diff = m / (sqrt(v) + epsilon);
  *model = model_val - learning_rate * (model_diff + weight_decay * model_val);
}

template<bool do_bias_correction, typename T>
__global__ void UpdateModelGpu(int64_t n, const float* learning_rate, T weight_decay, T beta1,
                               T beta2, T epsilon, T* beta1_t, T* beta2_t, const T* model_diff,
                               T* model, T* m, T* v) {
  float lr;
  if (do_bias_correction) {
    lr = *learning_rate * sqrt(1 - *beta2_t) / (1 - *beta1_t);
  } else {
    lr = *learning_rate;
  }
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T model_diff_val = model_diff[i];
    T m_val = m[i];
    UpdateMomentEstimate<1>(beta1, model_diff_val, &m_val);
    m[i] = m_val;
    T v_val = v[i];
    UpdateMomentEstimate<2>(beta2, model_diff_val, &v_val);
    v[i] = v_val;
    UpdateModel(lr, weight_decay, epsilon, model + i, m_val, v_val);
  }
}

template<typename T>
__global__ void UpdateBetaTGpu(const T beta1, const T beta2, T* beta1_t, T* beta2_t) {
  *beta1_t *= beta1;
  *beta2_t *= beta2;
}

}  // namespace

template<typename T>
class AdamMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T weight_decay,
                          T beta1, T beta2, T epsilon, bool do_bias_correction,
                          const int64_t* train_step, T* beta1_t, T* beta2_t, const T* model_diff,
                          T* model, T* m, T* v) {
    if (do_bias_correction) {
      UpdateModelGpu<true, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              n, learning_rate, weight_decay, beta1, beta2, epsilon, beta1_t, beta2_t, model_diff,
              model, m, v);
      UpdateBetaTGpu<T><<<1, 1, 0, ctx->cuda_stream()>>>(beta1, beta2, beta1_t, beta2_t);
    } else {
      UpdateModelGpu<false, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              n, learning_rate, weight_decay, beta1, beta2, epsilon, beta1_t, beta2_t, model_diff,
              model, m, v);
    }
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class AdamMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
