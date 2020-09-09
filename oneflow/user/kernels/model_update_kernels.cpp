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

template<DeviceType device_type, typename T, typename G>
class SGDUpdateKernel final : public user_op::OpKernel {
 public:
  SGDUpdateKernel() = default;
  ~SGDUpdateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    const auto scale = ctx->Attr<float>("scale");
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    SGDUpdateKernelUtil<device_type, T, G>::Update(
        ctx->device_ctx(), model->shape().elem_cnt(), scale, l1, l2, weight_decay,
        learning_rate->dptr<float>(), model_diff->dptr<G>(), model->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_SGD_UPDATE_KERNEL(device, dtype, gtype)                                 \
  REGISTER_USER_KERNEL("sgd_update")                                                     \
      .SetCreateFn<SGDUpdateKernel<device, dtype, gtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_SGD_UPDATE_KERNEL(DeviceType::kCPU, float, float);
REGISTER_SGD_UPDATE_KERNEL(DeviceType::kCPU, double, double);
REGISTER_SGD_UPDATE_KERNEL(DeviceType::kGPU, float, float16);
REGISTER_SGD_UPDATE_KERNEL(DeviceType::kGPU, float, float);
REGISTER_SGD_UPDATE_KERNEL(DeviceType::kGPU, double, double);

template<DeviceType device_type, typename T, typename G>
class MomentumUpdateKernel final : public user_op::OpKernel {
 public:
  MomentumUpdateKernel() = default;
  ~MomentumUpdateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* momentum = ctx->Tensor4ArgNameAndIndex("momentum", 0);
    const auto scale = ctx->Attr<float>("scale");
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");
    const auto beta = ctx->Attr<float>("beta");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    MomentumUpdateKernelUtil<device_type, T, G>::Update(
        ctx->device_ctx(), model->shape().elem_cnt(), scale, l1, l2, beta, weight_decay,
        learning_rate->dptr<float>(), model_diff->dptr<G>(), model->mut_dptr<T>(),
        momentum->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MOMENTUM_UPDATE_KERNEL(device, dtype, gtype)                            \
  REGISTER_USER_KERNEL("momentum_update")                                                \
      .SetCreateFn<MomentumUpdateKernel<device, dtype, gtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_MOMENTUM_UPDATE_KERNEL(DeviceType::kCPU, float, float);
REGISTER_MOMENTUM_UPDATE_KERNEL(DeviceType::kCPU, double, double);
REGISTER_MOMENTUM_UPDATE_KERNEL(DeviceType::kGPU, float, float16);
REGISTER_MOMENTUM_UPDATE_KERNEL(DeviceType::kGPU, float, float);
REGISTER_MOMENTUM_UPDATE_KERNEL(DeviceType::kGPU, double, double);

template<DeviceType device_type, typename T, typename G>
class AdamUpdateKernel final : public user_op::OpKernel {
 public:
  AdamUpdateKernel() = default;
  ~AdamUpdateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* m = ctx->Tensor4ArgNameAndIndex("m", 0);
    user_op::Tensor* v = ctx->Tensor4ArgNameAndIndex("v", 0);
    const auto scale = ctx->Attr<float>("scale");
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");
    const auto beta1 = ctx->Attr<float>("beta1");
    const auto beta2 = ctx->Attr<float>("beta2");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto do_bias_correction = ctx->Attr<bool>("do_bias_correction");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    T* beta1_t_ptr = nullptr;
    T* beta2_t_ptr = nullptr;
    if (do_bias_correction) {
      user_op::Tensor* beta1_t = ctx->Tensor4ArgNameAndIndex("beta1_t", 0);
      beta1_t_ptr = beta1_t->mut_dptr<T>();
      user_op::Tensor* beta2_t = ctx->Tensor4ArgNameAndIndex("beta2_t", 0);
      beta2_t_ptr = beta2_t->mut_dptr<T>();
    }
    AdamUpdateKernelUtil<device_type, T, G>::Update(
        ctx->device_ctx(), model->shape().elem_cnt(), scale, l1, l2, beta1, beta2, epsilon,
        do_bias_correction, weight_decay, learning_rate->dptr<float>(), model_diff->dptr<G>(),
        model->mut_dptr<T>(), m->mut_dptr<T>(), v->mut_dptr<T>(), beta1_t_ptr, beta2_t_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_ADAM_UPDATE_KERNEL(device, dtype, gtype)                                \
  REGISTER_USER_KERNEL("adam_update")                                                    \
      .SetCreateFn<AdamUpdateKernel<device, dtype, gtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_ADAM_UPDATE_KERNEL(DeviceType::kCPU, float, float);
REGISTER_ADAM_UPDATE_KERNEL(DeviceType::kCPU, double, double);
REGISTER_ADAM_UPDATE_KERNEL(DeviceType::kGPU, float, float16);
REGISTER_ADAM_UPDATE_KERNEL(DeviceType::kGPU, float, float);
REGISTER_ADAM_UPDATE_KERNEL(DeviceType::kGPU, double, double);

}  // namespace

}  // namespace oneflow
