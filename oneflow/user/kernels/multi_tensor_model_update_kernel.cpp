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
#include "oneflow/user/kernels/multi_tensor_model_update_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename G>
class MultiTensorSGDUpdateKernel final : public user_op::OpKernel,
                                         public user_op::CudaGraphSupport {
 public:
  MultiTensorSGDUpdateKernel() = default;
  ~MultiTensorSGDUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int64_t n_tensor = ctx->input_size("model");
    const double scale = ctx->Attr<double>("scale");
    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");
    const float weight_decay = ctx->Attr<float>("weight_decay");
    const float* learning_rate_ptr = nullptr;
    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    const float lr_scale = ctx->Attr<float>("learning_rate_scale");

    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), ctx->Tensor4ArgNameAndIndex("model", 0)->data_type());
      CHECK_EQ(scale_by_tensor->shape_view().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape_view().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }

    TensorTupleParams<2> tensor_tuple_params{};
    int32_t count = 0;
    int32_t total_elem_cnt = 0;
    for (int tensor_idx = 0; tensor_idx < n_tensor; tensor_idx++) {
      tensor_tuple_params.ptr[0][count] =
          (ctx->Tensor4ArgNameAndIndex("model", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[1][count] =
          (ctx->Tensor4ArgNameAndIndex("model_diff", tensor_idx))->mut_dptr();

      const int64_t tensor_elem_cnt =
          ctx->Tensor4ArgNameAndIndex("model", tensor_idx)->shape_view().elem_cnt();
      tensor_tuple_params.sizes[count] = tensor_elem_cnt;

      count += 1;
      total_elem_cnt += tensor_elem_cnt;
      if (count == kMaxTuples || tensor_idx == n_tensor - 1) {
        MultiTensorSGDUpdateKernelUtil<device_type, T, G>::Update(
            ctx->stream(), total_elem_cnt, count, static_cast<T>(scale), l1, l2, weight_decay,
            learning_rate_val, lr_scale, learning_rate_ptr, scale_by_ptr, skip_if_ptr,
            tensor_tuple_params);
        count = 0;
        total_elem_cnt = 0;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(device, dtype, gtype)              \
  REGISTER_USER_KERNEL("multi_tensor_sgd_update")                                         \
      .SetCreateFn<MultiTensorSGDUpdateKernel<device, dtype, gtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                               \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

#ifdef WITH_CUDA
REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(DeviceType::kCUDA, float, float16);
REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(DeviceType::kCUDA, float, float);
REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(DeviceType::kCUDA, double, double);
#endif

template<DeviceType device_type, typename T, typename G>
class MultiTensorMomentumUpdateKernel final : public user_op::OpKernel,
                                              public user_op::CudaGraphSupport {
 public:
  MultiTensorMomentumUpdateKernel() = default;
  ~MultiTensorMomentumUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int64_t n_tensor = ctx->input_size("model");
    const double scale = ctx->Attr<double>("scale");
    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");
    const float weight_decay = ctx->Attr<float>("weight_decay");
    const float* learning_rate_ptr = nullptr;
    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    const float lr_scale = ctx->Attr<float>("learning_rate_scale");
    const float momentum = ctx->Attr<float>("momentum");
    const float dampening = ctx->Attr<float>("dampening");
    const bool nesterov = ctx->Attr<bool>("nesterov");
    const bool maximize = ctx->Attr<bool>("maximize");

    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), ctx->Tensor4ArgNameAndIndex("model", 0)->data_type());
      CHECK_EQ(scale_by_tensor->shape_view().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape_view().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }

    TensorTupleParams<3> tensor_tuple_params{};
    int32_t count = 0;
    int32_t total_elem_cnt = 0;
    for (int tensor_idx = 0; tensor_idx < n_tensor; tensor_idx++) {
      tensor_tuple_params.ptr[0][count] =
          (ctx->Tensor4ArgNameAndIndex("model", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[1][count] =
          (ctx->Tensor4ArgNameAndIndex("model_diff", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[2][count] =
          (ctx->Tensor4ArgNameAndIndex("momentum_buf", tensor_idx))->mut_dptr();

      const int64_t tensor_elem_cnt =
          ctx->Tensor4ArgNameAndIndex("model", tensor_idx)->shape_view().elem_cnt();
      tensor_tuple_params.sizes[count] = tensor_elem_cnt;

      count += 1;
      total_elem_cnt += tensor_elem_cnt;
      if (count == kMaxTuples || tensor_idx == n_tensor - 1) {
        MultiTensorMomentumUpdateKernelUtil<device_type, T, G>::Update(
            ctx->stream(), total_elem_cnt, count, static_cast<T>(scale), l1, l2, weight_decay,
            learning_rate_val, lr_scale, learning_rate_ptr, scale_by_ptr, skip_if_ptr, momentum,
            dampening, nesterov, maximize, tensor_tuple_params);
        count = 0;
        total_elem_cnt = 0;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MULTI_TENSOR_UPDATE_MOMENTUM_UPDATE_KERNEL(device, dtype, gtype)              \
  REGISTER_USER_KERNEL("multi_tensor_momentum_update")                                         \
      .SetCreateFn<MultiTensorMomentumUpdateKernel<device, dtype, gtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                    \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value)      \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value) \
                       && (user_op::HobDataType("momentum_buf", 0) == GetDataType<gtype>::value));

#ifdef WITH_CUDA
REGISTER_MULTI_TENSOR_UPDATE_MOMENTUM_UPDATE_KERNEL(DeviceType::kCUDA, float, float16);
REGISTER_MULTI_TENSOR_UPDATE_MOMENTUM_UPDATE_KERNEL(DeviceType::kCUDA, float, float);
REGISTER_MULTI_TENSOR_UPDATE_MOMENTUM_UPDATE_KERNEL(DeviceType::kCUDA, double, double);
#endif

template<DeviceType device_type, typename T, typename G>
class MultiTensorAdamUpdateKernel final : public user_op::OpKernel,
                                          public user_op::CudaGraphSupport {
 public:
  MultiTensorAdamUpdateKernel() = default;
  ~MultiTensorAdamUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int64_t n_tensor = ctx->input_size("model");
    const auto scale = ctx->Attr<double>("scale");
    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");

    const float beta1 = ctx->Attr<float>("beta1");
    const float beta2 = ctx->Attr<float>("beta2");
    const float epsilon = ctx->Attr<float>("epsilon");
    const float weight_decay = ctx->Attr<float>("weight_decay");

    const bool amsgrad = ctx->Attr<bool>("amsgrad");
    const bool do_bias_correction = ctx->Attr<bool>("do_bias_correction");
    if (amsgrad) { UNIMPLEMENTED() << "Multi Tensor Adam Update do not support amsgrad = True. "; }

    const float* learning_rate_ptr = nullptr;
    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    const float lr_scale = ctx->Attr<float>("learning_rate_scale");

    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }

    const float bias_correction1_val = ctx->Attr<float>("bias_correction1_val");
    const float* bias_correction1_ptr = nullptr;
    if (ctx->has_input("bias_correction1", 0)) {
      const user_op::Tensor* bias_correction1 = ctx->Tensor4ArgNameAndIndex("bias_correction1", 0);
      CHECK_EQ(bias_correction1->shape_view().elem_cnt(),
               1);  // Just for Lazy Optional Input Check.
      bias_correction1_ptr = bias_correction1->dptr<float>();
    }

    const float bias_correction2_val = ctx->Attr<float>("bias_correction2_val");
    const float* bias_correction2_ptr = nullptr;
    if (ctx->has_input("bias_correction2", 0)) {
      const user_op::Tensor* bias_correction2 = ctx->Tensor4ArgNameAndIndex("bias_correction2", 0);
      CHECK_EQ(bias_correction2->shape_view().elem_cnt(),
               1);  // Just for Lazy Optional Input Check.
      bias_correction2_ptr = bias_correction2->dptr<float>();
    }

    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), ctx->Tensor4ArgNameAndIndex("model", 0)->data_type());
      CHECK_EQ(scale_by_tensor->shape_view().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape_view().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }

    TensorTupleParams<4> tensor_tuple_params{};
    int32_t count = 0;
    int32_t total_elem_cnt = 0;
    for (int tensor_idx = 0; tensor_idx < n_tensor; tensor_idx++) {
      tensor_tuple_params.ptr[0][count] =
          (ctx->Tensor4ArgNameAndIndex("model", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[1][count] =
          (ctx->Tensor4ArgNameAndIndex("model_diff", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[2][count] =
          (ctx->Tensor4ArgNameAndIndex("m", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[3][count] =
          (ctx->Tensor4ArgNameAndIndex("v", tensor_idx))->mut_dptr();
      const int64_t tensor_elem_cnt =
          ctx->Tensor4ArgNameAndIndex("model", tensor_idx)->shape_view().elem_cnt();
      tensor_tuple_params.sizes[count] = tensor_elem_cnt;

      count += 1;
      total_elem_cnt += tensor_elem_cnt;
      if (count == kMaxTuples || tensor_idx == n_tensor - 1) {
        MultiTensorAdamUpdateKernelUtil<device_type, T, G>::Update(
            ctx->stream(), total_elem_cnt, count, static_cast<T>(scale), l1, l2, beta1, beta2,
            epsilon, weight_decay, amsgrad, do_bias_correction, learning_rate_val,
            bias_correction1_val, bias_correction2_val, lr_scale, learning_rate_ptr, scale_by_ptr,
            skip_if_ptr, bias_correction1_ptr, bias_correction2_ptr, tensor_tuple_params);
        count = 0;
        total_elem_cnt = 0;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MULTI_TENSOR_UPDATE_ADAM_UPDATE_KERNEL(device, dtype, gtype)             \
  REGISTER_USER_KERNEL("multi_tensor_adam_update")                                        \
      .SetCreateFn<MultiTensorAdamUpdateKernel<device, dtype, gtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                               \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

#ifdef WITH_CUDA
REGISTER_MULTI_TENSOR_UPDATE_ADAM_UPDATE_KERNEL(DeviceType::kCUDA, float, float16);
REGISTER_MULTI_TENSOR_UPDATE_ADAM_UPDATE_KERNEL(DeviceType::kCUDA, float, float);
REGISTER_MULTI_TENSOR_UPDATE_ADAM_UPDATE_KERNEL(DeviceType::kCUDA, double, double);
#endif

template<DeviceType device_type, typename T, typename G>
class MultiTensorSGDUpdateWithCastKernel final : public user_op::OpKernel,
                                                 public user_op::CudaGraphSupport {
 public:
  MultiTensorSGDUpdateWithCastKernel() = default;
  ~MultiTensorSGDUpdateWithCastKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int64_t n_tensor = ctx->input_size("model");
    const double scale = ctx->Attr<double>("scale");
    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");
    const float weight_decay = ctx->Attr<float>("weight_decay");
    const float* learning_rate_ptr = nullptr;
    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    const float lr_scale = ctx->Attr<float>("learning_rate_scale");

    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), ctx->Tensor4ArgNameAndIndex("model", 0)->data_type());
      CHECK_EQ(scale_by_tensor->shape_view().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape_view().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }

    TensorTupleParams<3> tensor_tuple_params{};
    int32_t count = 0;
    int32_t total_elem_cnt = 0;
    for (int tensor_idx = 0; tensor_idx < n_tensor; tensor_idx++) {
      tensor_tuple_params.ptr[0][count] =
          (ctx->Tensor4ArgNameAndIndex("model", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[1][count] =
          (ctx->Tensor4ArgNameAndIndex("model_diff", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[2][count] =
          (ctx->Tensor4ArgNameAndIndex("model_copy", tensor_idx))->mut_dptr();

      const int64_t tensor_elem_cnt =
          ctx->Tensor4ArgNameAndIndex("model", tensor_idx)->shape_view().elem_cnt();
      tensor_tuple_params.sizes[count] = tensor_elem_cnt;

      count += 1;
      total_elem_cnt += tensor_elem_cnt;
      if (count == kMaxTuples || tensor_idx == n_tensor - 1) {
        MultiTensorSGDUpdateWithCastKernelUtil<device_type, T, G>::Update(
            ctx->stream(), total_elem_cnt, count, static_cast<T>(scale), l1, l2, weight_decay,
            learning_rate_val, lr_scale, learning_rate_ptr, scale_by_ptr, skip_if_ptr,
            tensor_tuple_params);
        count = 0;
        total_elem_cnt = 0;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_WITH_CAST_KERNEL(device, dtype, gtype)         \
  REGISTER_USER_KERNEL("multi_tensor_sgd_update_with_cast")                                    \
      .SetCreateFn<MultiTensorSGDUpdateWithCastKernel<device, dtype, gtype>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                    \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value)      \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value) \
                       && (user_op::HobDataType("model_copy", 0) == GetDataType<float16>::value));

#ifdef WITH_CUDA
REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_WITH_CAST_KERNEL(DeviceType::kCUDA, float, float);
REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_WITH_CAST_KERNEL(DeviceType::kCUDA, float, float16);
#endif

template<DeviceType device_type, typename T, typename G>
class MultiTensorMomentumUpdateWithCastKernel final : public user_op::OpKernel,
                                                      public user_op::CudaGraphSupport {
 public:
  MultiTensorMomentumUpdateWithCastKernel() = default;
  ~MultiTensorMomentumUpdateWithCastKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int64_t n_tensor = ctx->input_size("model");
    const double scale = ctx->Attr<double>("scale");
    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");
    const float weight_decay = ctx->Attr<float>("weight_decay");
    const float* learning_rate_ptr = nullptr;
    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    const float lr_scale = ctx->Attr<float>("learning_rate_scale");
    const float momentum = ctx->Attr<float>("momentum");
    const float dampening = ctx->Attr<float>("dampening");
    const bool nesterov = ctx->Attr<float>("nesterov");
    const bool maximize = ctx->Attr<float>("maximize");

    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), ctx->Tensor4ArgNameAndIndex("model", 0)->data_type());
      CHECK_EQ(scale_by_tensor->shape_view().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape_view().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }

    TensorTupleParams<4> tensor_tuple_params{};
    int32_t count = 0;
    int32_t total_elem_cnt = 0;
    for (int tensor_idx = 0; tensor_idx < n_tensor; tensor_idx++) {
      tensor_tuple_params.ptr[0][count] =
          (ctx->Tensor4ArgNameAndIndex("model", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[1][count] =
          (ctx->Tensor4ArgNameAndIndex("model_diff", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[2][count] =
          (ctx->Tensor4ArgNameAndIndex("momentum_buf", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[3][count] =
          (ctx->Tensor4ArgNameAndIndex("model_copy", tensor_idx))->mut_dptr();

      const int64_t tensor_elem_cnt =
          ctx->Tensor4ArgNameAndIndex("model", tensor_idx)->shape_view().elem_cnt();
      tensor_tuple_params.sizes[count] = tensor_elem_cnt;

      count += 1;
      total_elem_cnt += tensor_elem_cnt;
      if (count == kMaxTuples || tensor_idx == n_tensor - 1) {
        MultiTensorMomentumUpdateWithCastKernelUtil<device_type, T, G>::Update(
            ctx->stream(), total_elem_cnt, count, static_cast<T>(scale), l1, l2, weight_decay,
            learning_rate_val, lr_scale, learning_rate_ptr, scale_by_ptr, skip_if_ptr, momentum,
            dampening, nesterov, maximize, tensor_tuple_params);
        count = 0;
        total_elem_cnt = 0;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MULTI_TENSOR_UPDATE_MOMENTUM_UPDATE_WITH_CAST_KERNEL(device, dtype, gtype)      \
  REGISTER_USER_KERNEL("multi_tensor_momentum_update_with_cast")                                 \
      .SetCreateFn<MultiTensorMomentumUpdateWithCastKernel<device, dtype, gtype>>()              \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                      \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value)        \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value)   \
                       && (user_op::HobDataType("momentum_buf", 0) == GetDataType<gtype>::value) \
                       && (user_op::HobDataType("model_copy", 0) == GetDataType<float16>::value));

#ifdef WITH_CUDA
REGISTER_MULTI_TENSOR_UPDATE_MOMENTUM_UPDATE_WITH_CAST_KERNEL(DeviceType::kCUDA, float, float);
REGISTER_MULTI_TENSOR_UPDATE_MOMENTUM_UPDATE_WITH_CAST_KERNEL(DeviceType::kCUDA, float, float16);
#endif

template<DeviceType device_type, typename T, typename G>
class MultiTensorAdamUpdateWithCastKernel final : public user_op::OpKernel,
                                                  public user_op::CudaGraphSupport {
 public:
  MultiTensorAdamUpdateWithCastKernel() = default;
  ~MultiTensorAdamUpdateWithCastKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int64_t n_tensor = ctx->input_size("model");
    const auto scale = ctx->Attr<double>("scale");
    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");

    const float beta1 = ctx->Attr<float>("beta1");
    const float beta2 = ctx->Attr<float>("beta2");
    const float epsilon = ctx->Attr<float>("epsilon");
    const float weight_decay = ctx->Attr<float>("weight_decay");

    const bool amsgrad = ctx->Attr<bool>("amsgrad");
    const bool do_bias_correction = ctx->Attr<bool>("do_bias_correction");
    if (amsgrad) { UNIMPLEMENTED() << "Multi Tensor Adam Update do not support amsgrad = True. "; }

    const float* learning_rate_ptr = nullptr;
    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    const float lr_scale = ctx->Attr<float>("learning_rate_scale");

    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }

    const float bias_correction1_val = ctx->Attr<float>("bias_correction1_val");
    const float* bias_correction1_ptr = nullptr;
    if (ctx->has_input("bias_correction1", 0)) {
      const user_op::Tensor* bias_correction1 = ctx->Tensor4ArgNameAndIndex("bias_correction1", 0);
      CHECK_EQ(bias_correction1->shape_view().elem_cnt(),
               1);  // Just for Lazy Optional Input Check.
      bias_correction1_ptr = bias_correction1->dptr<float>();
    }

    const float bias_correction2_val = ctx->Attr<float>("bias_correction2_val");
    const float* bias_correction2_ptr = nullptr;
    if (ctx->has_input("bias_correction2", 0)) {
      const user_op::Tensor* bias_correction2 = ctx->Tensor4ArgNameAndIndex("bias_correction2", 0);
      CHECK_EQ(bias_correction2->shape_view().elem_cnt(),
               1);  // Just for Lazy Optional Input Check.
      bias_correction2_ptr = bias_correction2->dptr<float>();
    }

    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), ctx->Tensor4ArgNameAndIndex("model", 0)->data_type());
      CHECK_EQ(scale_by_tensor->shape_view().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape_view().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }

    TensorTupleParams<5> tensor_tuple_params{};
    int32_t count = 0;
    int32_t total_elem_cnt = 0;
    for (int tensor_idx = 0; tensor_idx < n_tensor; tensor_idx++) {
      tensor_tuple_params.ptr[0][count] =
          (ctx->Tensor4ArgNameAndIndex("model", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[1][count] =
          (ctx->Tensor4ArgNameAndIndex("model_diff", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[2][count] =
          (ctx->Tensor4ArgNameAndIndex("m", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[3][count] =
          (ctx->Tensor4ArgNameAndIndex("v", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[4][count] =
          (ctx->Tensor4ArgNameAndIndex("model_copy", tensor_idx))->mut_dptr();
      const int64_t tensor_elem_cnt =
          ctx->Tensor4ArgNameAndIndex("model", tensor_idx)->shape_view().elem_cnt();
      tensor_tuple_params.sizes[count] = tensor_elem_cnt;

      count += 1;
      total_elem_cnt += tensor_elem_cnt;
      if (count == kMaxTuples || tensor_idx == n_tensor - 1) {
        MultiTensorAdamUpdateWithCastKernelUtil<device_type, T, G>::Update(
            ctx->stream(), total_elem_cnt, count, static_cast<T>(scale), l1, l2, beta1, beta2,
            epsilon, weight_decay, amsgrad, do_bias_correction, learning_rate_val,
            bias_correction1_val, bias_correction2_val, lr_scale, learning_rate_ptr, scale_by_ptr,
            skip_if_ptr, bias_correction1_ptr, bias_correction2_ptr, tensor_tuple_params);
        count = 0;
        total_elem_cnt = 0;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MULTI_TENSOR_UPDATE_ADAM_UPDATE_WITH_CAST_KERNEL(device, dtype, gtype)        \
  REGISTER_USER_KERNEL("multi_tensor_adam_update_with_cast")                                   \
      .SetCreateFn<MultiTensorAdamUpdateWithCastKernel<device, dtype, gtype>>()                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                    \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value)      \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value) \
                       && (user_op::HobDataType("model_copy", 0) == GetDataType<float16>::value));

#ifdef WITH_CUDA
REGISTER_MULTI_TENSOR_UPDATE_ADAM_UPDATE_WITH_CAST_KERNEL(DeviceType::kCUDA, float, float);
REGISTER_MULTI_TENSOR_UPDATE_ADAM_UPDATE_WITH_CAST_KERNEL(DeviceType::kCUDA, float, float16);
#endif

template<DeviceType device_type, typename T>
class MultiTensorYoloV5WeightUpdateKernel final : public user_op::OpKernel,
                                                  public user_op::CudaGraphSupport {
 public:
  MultiTensorYoloV5WeightUpdateKernel() = default;
  ~MultiTensorYoloV5WeightUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int64_t n_tensor = ctx->input_size("model");
    const float d = ctx->Attr<float>("d");

    TensorTupleParams<2> tensor_tuple_params{};
    int32_t count = 0;
    int32_t total_elem_cnt = 0;
    for (int tensor_idx = 0; tensor_idx < n_tensor; tensor_idx++) {
      tensor_tuple_params.ptr[0][count] =
          (ctx->Tensor4ArgNameAndIndex("model", tensor_idx))->mut_dptr();
      tensor_tuple_params.ptr[1][count] =
          (ctx->Tensor4ArgNameAndIndex("model_update", tensor_idx))->mut_dptr();
      const int64_t tensor_elem_cnt =
          ctx->Tensor4ArgNameAndIndex("model", tensor_idx)->shape_view().elem_cnt();
      tensor_tuple_params.sizes[count] = tensor_elem_cnt;

      count += 1;
      total_elem_cnt += tensor_elem_cnt;
      if (count == kMaxTuples || tensor_idx == n_tensor - 1) {
        MultiTensorYoloV5WeightUpdateKernelUtil<device_type, T>::Update(
            ctx->stream(), total_elem_cnt, count, d, tensor_tuple_params);
        count = 0;
        total_elem_cnt = 0;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MULTI_TENSOR_YOLOV5_WEIGHT_UPDATE_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("multi_tensor_yolov5_weight_update")              \
      .SetCreateFn<MultiTensorYoloV5WeightUpdateKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)              \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_MULTI_TENSOR_YOLOV5_WEIGHT_UPDATE_KERNEL(DeviceType::kCUDA, float);
#endif

}  // namespace

}  // namespace oneflow
