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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/model_update_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename G>
__global__ void SGDUpdateGpu(int64_t n, float scale, float l1, float l2, float weight_decay,
                             const float* learning_rate, const G* model_diff, T* model) {
  const T lr = *learning_rate;
  CUDA_1D_KERNEL_LOOP(i, n) {
    SGDUpdateFunctor<T, G>()(model_diff + i, model + i, scale, l1, l2, weight_decay, lr);
  }
}

}  // namespace

template<typename T, typename G>
struct SGDUpdateKernelUtil<DeviceType::kGPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float weight_decay,
                     const float* learning_rate, const G* model_diff, T* model);
};

template<typename T, typename G>
void SGDUpdateKernelUtil<DeviceType::kGPU, T, G>::Update(DeviceCtx* ctx, int64_t n, float scale,
                                                         float l1, float l2, float weight_decay,
                                                         const float* learning_rate,
                                                         const G* model_diff, T* model) {
  SGDUpdateGpu<T, G><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, scale, l1, l2, weight_decay, learning_rate, model_diff, model);
}

template<typename T>
struct SGDUpdateKernelUtil<DeviceType::kGPU, T, float16> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float weight_decay,
                     const float* learning_rate, const float16* model_diff, T* model);
};

template<typename T>
void SGDUpdateKernelUtil<DeviceType::kGPU, T, float16>::Update(
    DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float weight_decay,
    const float* learning_rate, const float16* model_diff, T* model) {
  SGDUpdateKernelUtil<DeviceType::kGPU, T, half>::Update(
      ctx, n, scale, l1, l2, weight_decay, learning_rate, reinterpret_cast<const half*>(model_diff),
      model);
}

template struct SGDUpdateKernelUtil<DeviceType::kGPU, double, double>;
template struct SGDUpdateKernelUtil<DeviceType::kGPU, float, float>;
template struct SGDUpdateKernelUtil<DeviceType::kGPU, float, float16>;

namespace {

template<typename T, typename G>
__global__ void MomentumUpdateGpu(int64_t n, float scale, float l1, float l2, float beta,
                                  float weight_decay, const float* learning_rate,
                                  const G* model_diff, T* model, T* momentum) {
  const T lr = *learning_rate;
  CUDA_1D_KERNEL_LOOP(i, n) {
    MomentumUpdateFunctor<T, G>()(model_diff + i, model + i, momentum + i, scale, l1, l2, beta,
                                  weight_decay, lr);
  }
}

}  // namespace

template<typename T, typename G>
struct MomentumUpdateKernelUtil<DeviceType::kGPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta,
                     float weight_decay, const float* learning_rate, const G* model_diff, T* model,
                     T* momentum);
};

template<typename T, typename G>
void MomentumUpdateKernelUtil<DeviceType::kGPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta, float weight_decay,
    const float* learning_rate, const G* model_diff, T* model, T* momentum) {
  MomentumUpdateGpu<T, G>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, scale, l1, l2, beta, weight_decay, learning_rate, model_diff, model, momentum);
}

template<typename T>
struct MomentumUpdateKernelUtil<DeviceType::kGPU, T, float16> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta,
                     float weight_decay, const float* learning_rate, const float16* model_diff,
                     T* model, T* momentum);
};

template<typename T>
void MomentumUpdateKernelUtil<DeviceType::kGPU, T, float16>::Update(
    DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta, float weight_decay,
    const float* learning_rate, const float16* model_diff, T* model, T* momentum) {
  MomentumUpdateKernelUtil<DeviceType::kGPU, T, half>::Update(
      ctx, n, scale, l1, l2, beta, weight_decay, learning_rate,
      reinterpret_cast<const half*>(model_diff), model, momentum);
}

template struct MomentumUpdateKernelUtil<DeviceType::kGPU, double, double>;
template struct MomentumUpdateKernelUtil<DeviceType::kGPU, float, float>;
template struct MomentumUpdateKernelUtil<DeviceType::kGPU, float, float16>;

namespace {

template<typename T, typename G>
__global__ void AdamUpdateGpu(int64_t n, float scale, float l1, float l2, float beta1, float beta2,
                              float epsilon, bool do_bias_correction, float weight_decay,
                              const float* learning_rate, const G* model_diff, T* model, T* m, T* v,
                              T* beta1_t, T* beta2_t) {
  float lr;
  if (do_bias_correction) {
    lr = *learning_rate * sqrt(1 - *beta2_t) / (1 - *beta1_t);
  } else {
    lr = *learning_rate;
  }
  CUDA_1D_KERNEL_LOOP(i, n) {
    AdamUpdateFunctor<T, G>()(model_diff + i, model + i, m + i, v + i, scale, l1, l2, beta1, beta2,
                              epsilon, weight_decay, lr);
  }
}

template<typename T>
__global__ void AdamUpdateBetaTGpu(const T beta1, const T beta2, T* beta1_t, T* beta2_t) {
  *beta1_t *= beta1;
  *beta2_t *= beta2;
}

}  // namespace

template<typename T, typename G>
struct AdamUpdateKernelUtil<DeviceType::kGPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, bool do_bias_correction, float weight_decay,
                     const float* learning_rate, const G* model_diff, T* model, T* m, T* v,
                     T* beta1_t, T* beta2_t);
};

template<typename T, typename G>
void AdamUpdateKernelUtil<DeviceType::kGPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1, float beta2,
    float epsilon, bool do_bias_correction, float weight_decay, const float* learning_rate,
    const G* model_diff, T* model, T* m, T* v, T* beta1_t, T* beta2_t) {
  AdamUpdateGpu<T, G><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, scale, l1, l2, beta1, beta2, epsilon, do_bias_correction, weight_decay, learning_rate,
      model_diff, model, m, v, beta1_t, beta2_t);
  if (do_bias_correction) {
    AdamUpdateBetaTGpu<T><<<1, 1, 0, ctx->cuda_stream()>>>(beta1, beta2, beta1_t, beta2_t);
  }
}

template<typename T>
struct AdamUpdateKernelUtil<DeviceType::kGPU, T, float16> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, bool do_bias_correction, float weight_decay,
                     const float* learning_rate, const float16* model_diff, T* model, T* m, T* v,
                     T* beta1_t, T* beta2_t);
};

template<typename T>
void AdamUpdateKernelUtil<DeviceType::kGPU, T, float16>::Update(
    DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1, float beta2,
    float epsilon, bool do_bias_correction, float weight_decay, const float* learning_rate,
    const float16* model_diff, T* model, T* m, T* v, T* beta1_t, T* beta2_t) {
  AdamUpdateKernelUtil<DeviceType::kGPU, T, half>::Update(
      ctx, n, scale, l1, l2, beta1, beta2, epsilon, do_bias_correction, weight_decay, learning_rate,
      reinterpret_cast<const half*>(model_diff), model, m, v, beta1_t, beta2_t);
}

template struct AdamUpdateKernelUtil<DeviceType::kGPU, float, float>;
template struct AdamUpdateKernelUtil<DeviceType::kGPU, double, double>;
template struct AdamUpdateKernelUtil<DeviceType::kGPU, float, float16>;

}  // namespace oneflow
