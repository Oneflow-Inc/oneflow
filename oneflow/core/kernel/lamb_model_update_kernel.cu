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
#include "oneflow/core/kernel/lamb_model_update_kernel.h"

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
  *moment = beta * (*moment) + (1 - beta) * PowUtil<power>::pow(model_diff);
}

template<typename T>
__device__ void CorrectModelDiff(T epsilon, const T* beta1_t, const T* beta2_t, const T m_val,
                                 const T v_val, T* model_diff) {
  *model_diff = (m_val / (1 - *beta1_t)) * rsqrt(v_val / (1 - *beta2_t) + epsilon);
}

template<typename T>
__device__ void UpdateModel(const float learning_rate, T weight_decay, const T model_diff_val,
                            T* model) {
  T model_val = *model;
  *model = model_val - learning_rate * (model_diff_val + weight_decay * model_val);
}

template<typename T>
__global__ void UpdateMomentEstimateGpu(int64_t n, T beta1, T beta2, T epsilon, const T* beta1_t,
                                        const T* beta2_t, T* model_diff, T* model, T* m, T* v) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    UpdateMomentEstimate<1>(beta1, model_diff[i], &m[i]);
    UpdateMomentEstimate<2>(beta2, model_diff[i], &v[i]);
    CorrectModelDiff(epsilon, beta1_t, beta2_t, m[i], v[i], &model_diff[i]);
  }
}

template<typename T>
__global__ void UpdateModelGpu(int64_t n, const float* learning_rate, T weight_decay,
                               const T* fw_buf, T* model_diff, T* model) {
  const float local_lr = fw_buf[0] / fw_buf[1] * (*learning_rate);
  CUDA_1D_KERNEL_LOOP(i, n) { UpdateModel(local_lr, weight_decay, model_diff[i], &model[i]); }
}

template<typename T>
__global__ void UpdateBeta(const float* learning_rate, const int64_t* train_step, T beta1, T beta2,
                           T* beta1_t, T* beta2_t) {
  if (*train_step != 0) {
    *beta1_t *= beta1;
    *beta2_t *= beta2;
  }
}

}  // namespace

template<typename T>
class LAMBMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T weight_decay,
                          T beta1, T beta2, T epsilon, const int64_t* train_step, T* beta1_t,
                          T* beta2_t, T* model_diff, T* model, T* m, T* v, T* fw_buf) {
    UpdateBeta<T><<<1, 1, 0, ctx->cuda_stream()>>>(learning_rate, train_step, beta1, beta2, beta1_t,
                                                   beta2_t);
    UpdateMomentEstimateGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, beta1, beta2, epsilon, beta1_t, beta2_t, model_diff, model, m, v);
    KernelUtil<DeviceType::kGPU, T>::Dot(ctx, n, model, 1, model, 1, &fw_buf[0]);
    KernelUtil<DeviceType::kGPU, T>::Dot(ctx, n, model_diff, 1, model_diff, 1, &fw_buf[1]);
    KernelUtil<DeviceType::kGPU, T>::Sqrt(ctx, 2, fw_buf, fw_buf);
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, learning_rate, weight_decay, fw_buf, model_diff, model);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class LAMBMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
