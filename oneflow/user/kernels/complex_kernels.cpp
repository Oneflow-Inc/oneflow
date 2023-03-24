/*
Copyright 2023 The OneFlow Authors. All rights reserved.

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
#include <cufft.h>
#endif  // WITH_CUDA

namespace oneflow {
namespace user_op {

template <DeviceType device, typename dtype_x, typename dtype_out>
class RealKernel final : public user_op::OpKernel{
 public:
  RealKernel() = default;
  ~RealKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out->shape_view().elem_cnt() == 0) { return; }
    RealFunctor<device, dtype_x, dtype_out>(ctx->stream(), x, out);
  }
};

#define REGISTER_REAL_KERNEL(device, dtype_x, dtype_out)                               \
  REGISTER_USER_KERNEL("real")                                \
      .SetCreateFn<RealKernel<device, dtype_x, dtype_out>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype_x>::value));

REGISTER_REAL_KERNEL(DeviceType::kCPU, std::complex<float>, float)
REGISTER_REAL_KERNEL(DeviceType::kCPU, std::complex<double>, double)
#ifdef WITH_CUDA
REGISTER_REAL_KERNEL(DeviceType::kCUDA, cufftComplex, float)
REGISTER_REAL_KERNEL(DeviceType::kCUDA, cufftComplexDouble, double)
#endif  // WITH_CUDA

template <DeviceType device, typename dtype_x, typename dtype_out>
class ImagKernel final : public user_op::OpKernel{
 public:
  ImagKernel() = default;
  ~ImagKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out->shape_view().elem_cnt() == 0) { return; }
    ImagFunctor<device, dtype_x, dtype_out>(ctx->stream(), x, out);
  }
};

#define REGISTER_IMAG_KERNEL(device, dtype_x, dtype_out)                               \
  REGISTER_USER_KERNEL("imag")                                \
      .SetCreateFn<ImagKernel<device, dtype_x, dtype_out>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype_x>::value));

REGISTER_IMAG_KERNEL(DeviceType::kCPU, std::complex<float>, float)
REGISTER_IMAG_KERNEL(DeviceType::kCPU, std::complex<double>, double)
#ifdef WITH_CUDA
REGISTER_IMAG_KERNEL(DeviceType::kCUDA, cufftComplex, float)
REGISTER_IMAG_KERNEL(DeviceType::kCUDA, cufftComplexDouble, double)
#endif  // WITH_CUDA

template <DeviceType device, typename dtype>
class ConjPhysicalKernel final : public user_op::OpKernel{
 public:
  ConjPhysicalKernel() = default;
  ~ConjPhysicalKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out->shape_view().elem_cnt() == 0) { return; }
    ConjPhysicalFunctor<device, dtype>(ctx->stream(), x, out);
  }
};

#define REGISTER_CONJ_PHYSICAL_KERNEL(device, dtype)                               \
  REGISTER_USER_KERNEL("conj_physical")                                \
      .SetCreateFn<ConjPhysicalKernel<device, dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_CONJ_PHYSICAL_KERNEL(DeviceType::kCPU, std::complex<float>)
REGISTER_CONJ_PHYSICAL_KERNEL(DeviceType::kCPU, std::complex<double>)
#ifdef WITH_CUDA
REGISTER_CONJ_PHYSICAL_KERNEL(DeviceType::kCUDA, cufftComplex)
REGISTER_CONJ_PHYSICAL_KERNEL(DeviceType::kCUDA, cufftComplexDouble)
#endif  // WITH_CUDA

}  // namespace user_op
}  // namespace oneflow
