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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/cuda/primitive/binary_functor.cuh"
#include "oneflow/user/kernels/elementwise_primitive_kernel.h"

namespace oneflow {

namespace ep {
namespace primitive {
namespace broadcast_elementwise_binary {

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kFastGeluFuseMul, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src x, Src m) const {
    // ref to UnaryOp::kFastGelu
    const Src half = static_cast<Src>(0.5);
    const Src one = static_cast<Src>(1);
    const Src tanh_in = alpha * (x + beta * x * x * x);
    return half * x * (one + tanh(tanh_in)) * m;
  }

 private:
  const Src alpha = static_cast<Src>(0.7978845608028654);
  const Src beta = static_cast<Src>(0.044714998453855515);
};

SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kFastGeluFuseMul);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kFastGeluFuseMul);

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

namespace cuda {

REGISTER_USER_KERNEL("fused_fast_gelu_mul")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "out", "in", "multiplier", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kFastGeluFuseMul, in->data_type(),
                out->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kFastGeluFuseMul, "out", "in"));

template<typename T>
struct FusedFastGeluMulGradFunctor {
  OF_DEVICE_FUNC FusedFastGeluMulGradFunctor() {}

  OF_DEVICE_FUNC T operator()(const T dy, const T x, const T m) const {
    // ref to BinaryOp::kFastGeluBackwardWithDyX
    const T one = static_cast<T>(1);
    const T half = static_cast<T>(0.5);
    const T pow3 = x * x * x;
    const T tanh_out = tanh(alpha * (x + beta * pow3));
    const T dtanh = alpha * (half * x + beta * static_cast<T>(1.5) * pow3);
    return dy * m * (half + half * tanh_out + dtanh * (one - tanh_out * tanh_out));
  }

 private:
  const T alpha = static_cast<T>(0.7978845608028654);
  const T beta = static_cast<T>(0.044714998453855515);
};

template<>
struct FusedFastGeluMulGradFunctor<half> {
  OF_DEVICE_FUNC FusedFastGeluMulGradFunctor() {}

  OF_DEVICE_FUNC half operator()(const half dy, const half x, const half m) const {
    return __float2half(float_functor(__half2float(dy), __half2float(x), __half2float(m)));
  }

 private:
  FusedFastGeluMulGradFunctor<float> float_functor;
};

template<>
struct FusedFastGeluMulGradFunctor<nv_bfloat16> {
  OF_DEVICE_FUNC FusedFastGeluMulGradFunctor() {}

  OF_DEVICE_FUNC nv_bfloat16 operator()(const nv_bfloat16 dy, const nv_bfloat16 x,
                                        const nv_bfloat16 m) const {
    return __float2bfloat16(
        float_functor(__bfloat162float(dy), __bfloat162float(x), __bfloat162float(m)));
  }

 private:
  FusedFastGeluMulGradFunctor<float> float_functor;
};

template<typename T>
class FusedFastGeluMulGradCudaKernel final : public user_op::OpKernel {
 public:
  FusedFastGeluMulGradCudaKernel() = default;
  ~FusedFastGeluMulGradCudaKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* out_diff = ctx->Tensor4ArgNameAndIndex("out_diff", 0);
    const auto* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto* multiplier = ctx->Tensor4ArgNameAndIndex("multiplier", 0);
    auto* in_diff = ctx->Tensor4ArgNameAndIndex("in_diff", 0);

    int64_t elem_cnt = in->shape_view().elem_cnt();
    OF_CUDA_CHECK((elementwise::Ternary<FusedFastGeluMulGradFunctor<T>, T, T, T, T>(
        FusedFastGeluMulGradFunctor<T>(), elem_cnt, in_diff->mut_dptr<T>(), out_diff->dptr<T>(),
        in->dptr<T>(), multiplier->dptr<T>(), ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_FAST_GELU_MUL_GRAD_CUDA_KERNEL(dtype)           \
  REGISTER_USER_KERNEL("fused_fast_gelu_mul_grad")                     \
      .SetCreateFn<FusedFastGeluMulGradCudaKernel<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out_diff", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_FAST_GELU_MUL_GRAD_CUDA_KERNEL(float)
REGISTER_FUSED_FAST_GELU_MUL_GRAD_CUDA_KERNEL(double)
REGISTER_FUSED_FAST_GELU_MUL_GRAD_CUDA_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_FUSED_FAST_GELU_MUL_GRAD_CUDA_KERNEL(nv_bfloat16)
#endif

}  // namespace cuda

}  // namespace oneflow
