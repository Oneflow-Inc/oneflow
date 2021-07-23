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

namespace oneflow {

namespace {

template<typename T>
__global__ void SmoothL1LossForward(const int64_t elem_cnt, const T* prediction, const T* label,
                                    const T beta, T* loss) {
  const T half_beta = static_cast<T>(0.5) * beta;
  const T point5_div_beta = static_cast<T>(0.5) / beta;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T abs_diff = std::abs(prediction[i] - label[i]);
    if (abs_diff < beta) {
      loss[i] = abs_diff * abs_diff * point5_div_beta;
    } else {
      loss[i] = abs_diff - half_beta;
    }
  }
}

template<typename T>
__global__ void SmoothL1LossBackward(const int64_t elem_cnt, const T* loss_grad,
                                     const T* prediction, const T* label, const T beta,
                                     T* prediction_grad) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T diff = prediction[i] - label[i];
    const T abs_diff = std::abs(diff);
    if (abs_diff < beta) {
      prediction_grad[i] = diff / beta * loss_grad[i];
    } else {
      prediction_grad[i] = ((diff > GetZeroVal<T>()) - (diff < GetZeroVal<T>())) * loss_grad[i];
    }
  }
}

}  // namespace

template<typename T>
class SmoothL1LossGPUKernel final : public user_op::OpKernel {
 public:
  SmoothL1LossGPUKernel() = default;
  ~SmoothL1LossGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const float beta = ctx->Attr<float>("beta");
    const user_op::Tensor* prediction_blob = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const T* prediction = prediction_blob->dptr<T>();
    const int64_t elem_cnt = prediction_blob->shape().elem_cnt();
    const T* label = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<T>();
    T* loss = ctx->Tensor4ArgNameAndIndex("loss", 0)->mut_dptr<T>();
    SmoothL1LossForward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           ctx->device_ctx()->cuda_stream()>>>(elem_cnt, prediction, label, beta, loss);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SMOOTH_L1_LOSS_GPU_KERNEL(dtype)         \
  REGISTER_USER_KERNEL("smooth_l1_loss")                  \
      .SetCreateFn<SmoothL1LossGPUKernel<dtype>>()        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("loss", 0) == GetDataType<dtype>::value));

REGISTER_SMOOTH_L1_LOSS_GPU_KERNEL(float)
REGISTER_SMOOTH_L1_LOSS_GPU_KERNEL(double)

template<typename T>
class SmoothL1LossGradGpuKernel final : public user_op::OpKernel {
 public:
  SmoothL1LossGradGpuKernel() = default;
  ~SmoothL1LossGradGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const float beta = ctx->Attr<float>("beta");
    const user_op::Tensor* prediction_blob = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const T* prediction = prediction_blob->dptr<T>();
    const int64_t elem_cnt = prediction_blob->shape().elem_cnt();
    const T* loss_grad = ctx->Tensor4ArgNameAndIndex("loss_grad", 0)->dptr<T>();
    const T* label = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<T>();
    T* prediction_grad = ctx->Tensor4ArgNameAndIndex("prediction_grad", 0)->mut_dptr<T>();
    SmoothL1LossBackward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                              ctx->device_ctx()->cuda_stream()>>>(elem_cnt, loss_grad, prediction,
                                                                  label, beta, prediction_grad);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SMOOTH_L1_LOSS_GRAD_GPU_KERNEL(dtype) \
  REGISTER_USER_KERNEL("smooth_l1_loss_grad")          \
      .SetCreateFn<SmoothL1LossGradGpuKernel<dtype>>() \
      .SetIsMatchedHob(                                \
          (user_op::HobDeviceTag() == "gpu")           \
          & (user_op::HobDataType("prediction_grad", 0) == GetDataType<dtype>::value));

REGISTER_SMOOTH_L1_LOSS_GRAD_GPU_KERNEL(float)
REGISTER_SMOOTH_L1_LOSS_GRAD_GPU_KERNEL(double)

}  // namespace oneflow
