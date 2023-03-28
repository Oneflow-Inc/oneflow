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
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/complex_kernels_util.h"
#include <complex>
#ifdef WITH_CUDA
#include <cuComplex.h>
#endif  // WITH_CUDA

namespace oneflow {
namespace user_op {

template<DeviceType device, typename dtype_x, typename dtype_out>
class RealKernel final : public user_op::OpKernel {
 public:
  RealKernel() = default;
  ~RealKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out_tensor->shape_view().elem_cnt() == 0) { return; }
    const dtype_x* x = x_tensor->dptr<dtype_x>();
    dtype_out* out = out_tensor->mut_dptr<dtype_out>();
    RealFunctor<device, dtype_x, dtype_out>()(ctx->stream(), x, out,
                                              out_tensor->shape_view().elem_cnt());
  }
};

#define REGISTER_REAL_KERNEL(device, dtype_x, dtype_out)     \
  REGISTER_USER_KERNEL("real")                               \
      .SetCreateFn<RealKernel<device, dtype_x, dtype_out>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)  \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype_x>::value));

REGISTER_REAL_KERNEL(DeviceType::kCPU, std::complex<float>, float)
REGISTER_REAL_KERNEL(DeviceType::kCPU, std::complex<double>, double)
#ifdef WITH_CUDA
REGISTER_REAL_KERNEL(DeviceType::kCUDA, cuComplex, float)
REGISTER_REAL_KERNEL(DeviceType::kCUDA, cuDoubleComplex, double)
#endif  // WITH_CUDA

template<DeviceType device, typename dtype_dout, typename dtype_dx>
class RealGradKernel final : public user_op::OpKernel {
 public:
  RealGradKernel() = default;
  ~RealGradKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dout_tensor = ctx->Tensor4ArgNameAndIndex("dout", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_tensor->shape_view().elem_cnt() == 0) { return; }
    const dtype_dout* dout = dout_tensor->dptr<dtype_dout>();
    dtype_dx* dx = dx_tensor->mut_dptr<dtype_dx>();
    RealGradFunctor<device, dtype_dout, dtype_dx>()(ctx->stream(), dout, dx,
                                                    dx_tensor->shape_view().elem_cnt());
  }
};

#define REGISTER_REAL_GRAD_KERNEL(device, dtype_dout, dtype_dx)    \
  REGISTER_USER_KERNEL("real_grad")                                \
      .SetCreateFn<RealGradKernel<device, dtype_dout, dtype_dx>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)        \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype_dx>::value));

REGISTER_REAL_GRAD_KERNEL(DeviceType::kCPU, float, std::complex<float>)
REGISTER_REAL_GRAD_KERNEL(DeviceType::kCPU, double, std::complex<double>)
#ifdef WITH_CUDA
REGISTER_REAL_GRAD_KERNEL(DeviceType::kCUDA, float, cuComplex)
REGISTER_REAL_GRAD_KERNEL(DeviceType::kCUDA, double, cuDoubleComplex)
#endif  // WITH_CUDA

template<DeviceType device, typename dtype_x, typename dtype_out>
class ImagKernel final : public user_op::OpKernel {
 public:
  ImagKernel() = default;
  ~ImagKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out_tensor->shape_view().elem_cnt() == 0) { return; }
    const dtype_x* x = x_tensor->dptr<dtype_x>();
    dtype_out* out = out_tensor->mut_dptr<dtype_out>();
    ImagFunctor<device, dtype_x, dtype_out>()(ctx->stream(), x, out,
                                              out_tensor->shape_view().elem_cnt());
  }
};

#define REGISTER_IMAG_KERNEL(device, dtype_x, dtype_out)     \
  REGISTER_USER_KERNEL("imag")                               \
      .SetCreateFn<ImagKernel<device, dtype_x, dtype_out>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)  \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype_x>::value));

REGISTER_IMAG_KERNEL(DeviceType::kCPU, std::complex<float>, float)
REGISTER_IMAG_KERNEL(DeviceType::kCPU, std::complex<double>, double)
#ifdef WITH_CUDA
REGISTER_IMAG_KERNEL(DeviceType::kCUDA, cuComplex, float)
REGISTER_IMAG_KERNEL(DeviceType::kCUDA, cuDoubleComplex, double)
#endif  // WITH_CUDA

template<DeviceType device, typename dtype_dout, typename dtype_dx>
class ImagGradKernel final : public user_op::OpKernel {
 public:
  ImagGradKernel() = default;
  ~ImagGradKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dout_tensor = ctx->Tensor4ArgNameAndIndex("dout", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_tensor->shape_view().elem_cnt() == 0) { return; }
    const dtype_dout* dout = dout_tensor->dptr<dtype_dout>();
    dtype_dx* dx = dx_tensor->mut_dptr<dtype_dx>();
    ImagGradFunctor<device, dtype_dout, dtype_dx>()(ctx->stream(), dout, dx,
                                                    dx_tensor->shape_view().elem_cnt());
  }
};

#define REGISTER_IMAG_GRAD_KERNEL(device, dtype_dout, dtype_dx)    \
  REGISTER_USER_KERNEL("imag_grad")                                \
      .SetCreateFn<ImagGradKernel<device, dtype_dout, dtype_dx>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)        \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype_dx>::value));

REGISTER_IMAG_GRAD_KERNEL(DeviceType::kCPU, float, std::complex<float>)
REGISTER_IMAG_GRAD_KERNEL(DeviceType::kCPU, double, std::complex<double>)
#ifdef WITH_CUDA
REGISTER_IMAG_GRAD_KERNEL(DeviceType::kCUDA, float, cuComplex)
REGISTER_IMAG_GRAD_KERNEL(DeviceType::kCUDA, double, cuDoubleComplex)
#endif  // WITH_CUDA

template<DeviceType device, typename dtype>
class ConjPhysicalKernel final : public user_op::OpKernel {
 public:
  ConjPhysicalKernel() = default;
  ~ConjPhysicalKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out_tensor->shape_view().elem_cnt() == 0) { return; }
    const dtype* x = x_tensor->dptr<dtype>();
    dtype* out = out_tensor->mut_dptr<dtype>();
    ConjPhysicalFunctor<device, dtype>()(ctx->stream(), x, out,
                                         out_tensor->shape_view().elem_cnt());
  }
};

#define REGISTER_CONJ_PHYSICAL_KERNEL(device, dtype)        \
  REGISTER_USER_KERNEL("conj_physical")                     \
      .SetCreateFn<ConjPhysicalKernel<device, dtype>>()     \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_CONJ_PHYSICAL_KERNEL(DeviceType::kCPU, std::complex<float>)
REGISTER_CONJ_PHYSICAL_KERNEL(DeviceType::kCPU, std::complex<double>)
#ifdef WITH_CUDA
REGISTER_CONJ_PHYSICAL_KERNEL(DeviceType::kCUDA, cuComplex)
REGISTER_CONJ_PHYSICAL_KERNEL(DeviceType::kCUDA, cuDoubleComplex)
#endif  // WITH_CUDA

}  // namespace user_op
}  // namespace oneflow
