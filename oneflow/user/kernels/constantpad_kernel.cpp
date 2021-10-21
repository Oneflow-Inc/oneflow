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
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/constantpad_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
T GetDtypeMatchedValue(double floating, int64_t integral);

template<>
float16 GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<float16>(floating);
}

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

namespace user_op {

template<DeviceType device_type, typename IN_T>
class ConstantPad1dKernel final : public OpKernel {
 public:
  ConstantPad1dKernel() = default;
  ~ConstantPad1dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const ShapeView& x_shape = x->shape();
    const ShapeView& y_shape = y->shape();
    CHECK_EQ(x->shape().NumAxes(), 3);
    const std::vector<int64_t>& padding = ctx->Attr<std::vector<int64_t>>("padding");
    CHECK_EQ(padding.size(), 2);
    const IN_T constant_value = GetDtypeMatchedValue<IN_T>(ctx->Attr<double>("floating_value"),
                                                           ctx->Attr<int64_t>("integral_value"));

    IN_T* dest = y->mut_dptr<IN_T>();
    const IN_T* src = x->dptr<IN_T>();
    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 3> index_helper(y_vector.data());

    ConstantPad1dFunctor<device_type, IN_T>()(ctx->device_ctx(), src, dest, index_helper, x_shape,
                                              y_shape, padding, constant_value);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename IN_T>
class ConstantPad1dGradKernel final : public OpKernel {
 public:
  ConstantPad1dGradKernel() = default;
  ~ConstantPad1dGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    CHECK_EQ(dy->shape().NumAxes(), 3);
    Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const ShapeView& dx_shape = dx->shape();
    const ShapeView& dy_shape = dy->shape();

    const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
    CHECK_EQ(padding.size(), 2);

    const IN_T* src = dy->dptr<IN_T>();
    IN_T* dest = dx->mut_dptr<IN_T>();
    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 3> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().Count(0) * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->device_ctx(), dest, 0, out_bytes_size);

    ConstantPad1dGradFunctor<device_type, IN_T>()(ctx->device_ctx(), src, dest, index_helper,
                                                  dy_shape, dx_shape, padding);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename IN_T>
class ConstantPad3dKernel final : public OpKernel {
 public:
  ConstantPad3dKernel() = default;
  ~ConstantPad3dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const ShapeView& x_shape = x->shape();
    const ShapeView& y_shape = y->shape();
    CHECK_EQ(x->shape().NumAxes(), 5);
    const std::vector<int64_t>& padding = ctx->Attr<std::vector<int64_t>>("padding");
    CHECK_EQ(padding.size(), 6);
    const IN_T constant_value = GetDtypeMatchedValue<IN_T>(ctx->Attr<double>("floating_value"),
                                                           ctx->Attr<int64_t>("integral_value"));

    IN_T* dest = y->mut_dptr<IN_T>();
    const IN_T* src = x->dptr<IN_T>();
    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 5> index_helper(y_vector.data());

    ConstantPad3dFunctor<device_type, IN_T>()(ctx->device_ctx(), src, dest, index_helper, x_shape,
                                              y_shape, padding, constant_value);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename IN_T>
class ConstantPad3dGradKernel final : public OpKernel {
 public:
  ConstantPad3dGradKernel() = default;
  ~ConstantPad3dGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    CHECK_EQ(dy->shape().NumAxes(), 5);
    Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const ShapeView& dx_shape = dx->shape();
    const ShapeView& dy_shape = dy->shape();

    const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
    CHECK_EQ(padding.size(), 6);

    const IN_T* src = dy->dptr<IN_T>();
    IN_T* dest = dx->mut_dptr<IN_T>();
    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 5> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().Count(0) * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->device_ctx(), dest, 0, out_bytes_size);

    ConstantPad3dGradFunctor<device_type, IN_T>()(ctx->device_ctx(), src, dest, index_helper,
                                                  dy_shape, dx_shape, padding);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONSTANT_PAD_KERNELS(device, dtype)                                    \
  REGISTER_USER_KERNEL("constant_pad1d")                                                \
      .SetCreateFn<ConstantPad1dKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));  \
  REGISTER_USER_KERNEL("constant_pad1d_grad")                                           \
      .SetCreateFn<ConstantPad1dGradKernel<device, dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("constant_pad3d")                                                \
      .SetCreateFn<ConstantPad3dKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));  \
  REGISTER_USER_KERNEL("constant_pad3d_grad")                                           \
      .SetCreateFn<ConstantPad3dGradKernel<device, dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#define REGISTER_CONSTANT_PAD_WITH_DEVICE(device) \
  REGISTER_CONSTANT_PAD_KERNELS(device, float)    \
  REGISTER_CONSTANT_PAD_KERNELS(device, double)   \
  REGISTER_CONSTANT_PAD_KERNELS(device, int32_t)

REGISTER_CONSTANT_PAD_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_CONSTANT_PAD_WITH_DEVICE(DeviceType::kGPU)
REGISTER_CONSTANT_PAD_KERNELS(DeviceType::kGPU, float16)
#endif

}  // namespace user_op
}  // namespace oneflow
