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
#include "oneflow/user/kernels/clip_by_value_kernel.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T>
T GetDtypeMatchedValue(double floating, int64_t integral);

template<>
float GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<float>(floating);
}

template<>
double GetDtypeMatchedValue(double floating, int64_t integral) {
  return floating;
}

template<>
int8_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<int8_t>(integral);
}

template<>
int32_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<int32_t>(integral);
}

template<>
int64_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return integral;
}

}  // namespace

template<typename T>
struct ClipKernelUtil<DeviceType::kCPU, T> {
  template<typename F>
  static void Forward(DeviceCtx* ctx, F clip_func, const int64_t n, const T* x, T* y) {
    FOR_RANGE(int64_t, i, 0, n) { y[i] = clip_func(x[i]); }
  }

  template<typename F>
  static void Backward(DeviceCtx* ctx, F clip_func, const int64_t n, const T* x, const T* dy,
                       T* dx) {
    FOR_RANGE(int64_t, i, 0, n) { dx[i] = clip_func(x[i], dy[i]); }
  }
};

template<DeviceType device_type, typename T>
class ClipByScalarKernel final : public user_op::OpKernel {
 public:
  ClipByScalarKernel() = default;
  ~ClipByScalarKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    double floating_min = ctx->Attr<double>("floating_min");
    int64_t integral_min = ctx->Attr<int64_t>("integral_min");
    double floating_max = ctx->Attr<double>("floating_max");
    int64_t integral_max = ctx->Attr<int64_t>("integral_max");
    ClipByMinMaxFunctor<T> clip_func(GetDtypeMatchedValue<T>(floating_min, integral_min),
                                     GetDtypeMatchedValue<T>(floating_max, integral_max));
    ClipKernelUtil<device_type, T>::Forward(ctx->device_ctx(), clip_func, y->shape().elem_cnt(),
                                            x->dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class ClipByScalarMinKernel final : public user_op::OpKernel {
 public:
  ClipByScalarMinKernel() = default;
  ~ClipByScalarMinKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    double floating_min = ctx->Attr<double>("floating_min");
    int64_t integral_min = ctx->Attr<int64_t>("integral_min");
    ClipByMinFunctor<T> clip_func(GetDtypeMatchedValue<T>(floating_min, integral_min));
    ClipKernelUtil<device_type, T>::Forward(ctx->device_ctx(), clip_func, y->shape().elem_cnt(),
                                            x->dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class ClipByScalarMaxKernel final : public user_op::OpKernel {
 public:
  ClipByScalarMaxKernel() = default;
  ~ClipByScalarMaxKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    double floating_max = ctx->Attr<double>("floating_max");
    int64_t integral_max = ctx->Attr<int64_t>("integral_max");
    ClipByMaxFunctor<T> clip_func(GetDtypeMatchedValue<T>(floating_max, integral_max));
    ClipKernelUtil<device_type, T>::Forward(ctx->device_ctx(), clip_func, y->shape().elem_cnt(),
                                            x->dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class ClipByScalarGradKernel final : public user_op::OpKernel {
 public:
  ClipByScalarGradKernel() = default;
  ~ClipByScalarGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    double floating_min = ctx->Attr<double>("floating_min");
    int64_t integral_min = ctx->Attr<int64_t>("integral_min");
    double floating_max = ctx->Attr<double>("floating_max");
    int64_t integral_max = ctx->Attr<int64_t>("integral_max");
    ClipByMinMaxGradFunctor<T> clip_func(GetDtypeMatchedValue<T>(floating_min, integral_min),
                                         GetDtypeMatchedValue<T>(floating_max, integral_max));
    ClipKernelUtil<device_type, T>::Backward(ctx->device_ctx(), clip_func, dx->shape().elem_cnt(),
                                             x->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class ClipByScalarMinGradKernel final : public user_op::OpKernel {
 public:
  ClipByScalarMinGradKernel() = default;
  ~ClipByScalarMinGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    double floating_min = ctx->Attr<double>("floating_min");
    int64_t integral_min = ctx->Attr<int64_t>("integral_min");
    ClipByMinGradFunctor<T> clip_func(GetDtypeMatchedValue<T>(floating_min, integral_min));
    ClipKernelUtil<device_type, T>::Backward(ctx->device_ctx(), clip_func, dx->shape().elem_cnt(),
                                             x->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class ClipByScalarMaxGradKernel final : public user_op::OpKernel {
 public:
  ClipByScalarMaxGradKernel() = default;
  ~ClipByScalarMaxGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    double floating_max = ctx->Attr<double>("floating_max");
    int64_t integral_max = ctx->Attr<int64_t>("integral_max");
    ClipByMaxGradFunctor<T> clip_func(GetDtypeMatchedValue<T>(floating_max, integral_max));
    ClipKernelUtil<device_type, T>::Backward(ctx->device_ctx(), clip_func, dx->shape().elem_cnt(),
                                             x->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CLIP_KERNEL(op_type_name, kernel_name, device_type_v, dtype)                   \
  REGISTER_USER_KERNEL(#op_type_name)                                                           \
      .SetCreateFn<kernel_name##Kernel<device_type_v, dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device_type_v)                               \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value))           \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });

#define REGISTER_CLIP_GRAD_KERNEL(op_type_name, kernel_name, device_type_v, dtype)              \
  REGISTER_USER_KERNEL(#op_type_name)                                                           \
      .SetCreateFn<kernel_name##GradKernel<device_type_v, dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device_type_v)                               \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

#define REGISTER_CLIP_KERNELS(device_type_v, dtype_pair)                                          \
  REGISTER_CLIP_KERNEL(clip_by_scalar, ClipByScalar, device_type_v, OF_PP_PAIR_FIRST(dtype_pair)) \
  REGISTER_CLIP_KERNEL(clip_by_scalar_min, ClipByScalarMin, device_type_v,                        \
                       OF_PP_PAIR_FIRST(dtype_pair))                                              \
  REGISTER_CLIP_KERNEL(clip_by_scalar_max, ClipByScalarMax, device_type_v,                        \
                       OF_PP_PAIR_FIRST(dtype_pair))                                              \
  REGISTER_CLIP_GRAD_KERNEL(clip_by_scalar_grad, ClipByScalar, device_type_v,                     \
                            OF_PP_PAIR_FIRST(dtype_pair))                                         \
  REGISTER_CLIP_GRAD_KERNEL(clip_by_scalar_min_grad, ClipByScalarMin, device_type_v,              \
                            OF_PP_PAIR_FIRST(dtype_pair))                                         \
  REGISTER_CLIP_GRAD_KERNEL(clip_by_scalar_max_grad, ClipByScalarMax, device_type_v,              \
                            OF_PP_PAIR_FIRST(dtype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CLIP_KERNELS, DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
