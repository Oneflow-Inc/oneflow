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

template<typename T, typename G>
struct SGDUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float weight_decay,
                     const float* learning_rate, const G* model_diff, T* model);
};

template<typename T, typename G>
void SGDUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(DeviceCtx* ctx, int64_t n, float scale,
                                                         float l1, float l2, float weight_decay,
                                                         const float* learning_rate,
                                                         const G* model_diff, T* model) {
  const T lr = *learning_rate;
  for (int64_t i = 0; i != n; ++i) {
    SGDUpdateFunctor<T, G>()(model_diff + i, model + i, scale, l1, l2, weight_decay, lr);
  }
}

template struct SGDUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct SGDUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename G>
struct MomentumUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta,
                     float weight_decay, const float* learning_rate, const G* model_diff, T* model,
                     T* momentum);
};

template<typename T, typename G>
void MomentumUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta, float weight_decay,
    const float* learning_rate, const G* model_diff, T* model, T* momentum) {
  const T lr = *learning_rate;
  for (int64_t i = 0; i != n; ++i) {
    MomentumUpdateFunctor<T, G>()(model_diff + i, model + i, momentum + i, scale, l1, l2, beta,
                                  weight_decay, lr);
  }
}

template struct MomentumUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct MomentumUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename G>
struct AdamUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, bool do_bias_correction, float weight_decay,
                     const float* learning_rate, const G* model_diff, T* model, T* m, T* v,
                     T* beta1_t, T* beta2_t);
};

template<typename T, typename G>
void AdamUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1, float beta2,
    float epsilon, bool do_bias_correction, float weight_decay, const float* learning_rate,
    const G* model_diff, T* model, T* m, T* v, T* beta1_t, T* beta2_t) {
  float lr;
  if (do_bias_correction) {
    lr = *learning_rate * std::sqrt(1 - *beta2_t) / (1 - *beta1_t);
    *beta1_t *= beta1;
    *beta2_t *= beta2;
  } else {
    lr = *learning_rate;
  }
  FOR_RANGE(int64_t, i, 0, n) {
    AdamUpdateFunctor<T, G>()(model_diff + i, model + i, m + i, v + i, scale, l1, l2, beta1, beta2,
                              epsilon, weight_decay, lr);
  }
}

template struct AdamUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct AdamUpdateKernelUtil<DeviceType::kCPU, double, double>;

}  // namespace oneflow
