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

namespace oneflow {
namespace cuda {

namespace fused_gelu {

OF_DEVICE_FUNC float TanhApprox(float x) {
#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  float r;
  asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  return r;
#else
  return tanhf(x);
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
}

template<typename T>
struct FusedFastGeluMulFunctor {
  static constexpr T alpha = static_cast<T>(0.7978845608028654);
  static constexpr T beta = static_cast<T>(0.044714998453855515);

  OF_DEVICE_FUNC FusedFastGeluMulFunctor() {}

  OF_DEVICE_FUNC T operator()(T x, T m) const {
    // ref to UnaryFunctor of kFastGelu
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);
    const T tanh_in = alpha * (x + beta * x * x * x);
    return half * x * (one + tanh(tanh_in)) * m;
  }
};

template<>
struct FusedFastGeluMulFunctor<half> {
  static constexpr float alpha = FusedFastGeluMulFunctor<float>::alpha;
  static constexpr float beta = FusedFastGeluMulFunctor<float>::beta;
  FusedFastGeluMulFunctor<float> float_functor;

  OF_DEVICE_FUNC FusedFastGeluMulFunctor() {}

  OF_DEVICE_FUNC half operator()(const half x, const half m) const {
#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
    const float tanh_in =
        __half2float(__float2half_rn(alpha) * (x + __float2half_rn(beta) * x * x * x));
    const float tanh_out = TanhApprox(tanh_in);
    return __float2half_rn(0.5F) * x * (__float2half_rn(1.0F) + __float2half_rn(tanh_out)) * m;
#else
    return static_cast<half>(float_functor(static_cast<float>(x), static_cast<float>(m)));
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  }

#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  __device__ void Apply2(half* y, const half* x, const half* m) const {
    const half2 x2 = *(reinterpret_cast<const half2*>(x));
    const float2 tanh_in = __half22float2(
        __hmul2(__float2half2_rn(alpha),
                __hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta), x2), x2), x2))));
    float2 tanh_out;
    tanh_out.x = TanhApprox(tanh_in.x);
    tanh_out.y = TanhApprox(tanh_in.y);
    const half2 m2 = *(reinterpret_cast<const half2*>(m));
    const half2 y2 = __hmul2(__hmul2(__hmul2(__float2half2_rn(0.5F), x2),
                                     __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out))),
                             m2);
    *reinterpret_cast<half2*>(y) = y2;
  }
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
};

#if CUDA_VERSION >= 11000

template<>
struct FusedFastGeluMulFunctor<nv_bfloat16> {
  FusedFastGeluMulFunctor<float> float_functor;

  OF_DEVICE_FUNC FusedFastGeluMulFunctor() {}

  OF_DEVICE_FUNC nv_bfloat16 operator()(const nv_bfloat16 x, const nv_bfloat16 m) const {
    return __float2bfloat16(float_functor(__bfloat162float(x), __bfloat162float(m)));
  }
};

#endif  // CUDA_VERSION >= 11000

}  // namespace fused_gelu

template<typename T>
class FusedFastGeluMulKernel final : public user_op::OpKernel {
 public:
  FusedFastGeluMulKernel() = default;
  ~FusedFastGeluMulKernel() override = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto* multiplier = ctx->Tensor4ArgNameAndIndex("multiplier", 0);
    auto* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    int64_t elem_cnt = in->shape_view().elem_cnt();
    OF_CUDA_CHECK((elementwise::Binary(fused_gelu::FusedFastGeluMulFunctor<T>(), elem_cnt,
                                       out->mut_dptr<T>(), in->dptr<T>(), multiplier->dptr<T>(),
                                       ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
  };
};

#define REGISTER_FUSED_FAST_GELU_MUL_CUDA_KERNEL(dtype)                \
  REGISTER_USER_KERNEL("fused_fast_gelu_mul")                          \
      .SetCreateFn<FusedFastGeluMulKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_FAST_GELU_MUL_CUDA_KERNEL(float)
REGISTER_FUSED_FAST_GELU_MUL_CUDA_KERNEL(double)
REGISTER_FUSED_FAST_GELU_MUL_CUDA_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_FUSED_FAST_GELU_MUL_CUDA_KERNEL(nv_bfloat16)
#endif

namespace fused_gelu {

template<typename T>
struct FusedFastGeluMulGradFunctor {
  static constexpr T alpha = static_cast<T>(0.7978845608028654);
  static constexpr T beta = static_cast<T>(0.044714998453855515);

  __device__ FusedFastGeluMulGradFunctor() {}

  __device__ void operator()(T& x_diff, T& m_diff, const T& dy, const T& x, const T& m) const {
    const T one = static_cast<T>(1);
    const T half = static_cast<T>(0.5);
    const T pow3 = x * x * x;
    const T tanh_in = alpha * (x + beta * pow3);
    const T tanh_out = tanh(alpha * (x + beta * pow3));
    // calc m_diff ref to UnaryFunctor of kFastGelu
    m_diff = half * x * (one + tanh(tanh_in)) * dy;
    // calc x_diff ref to BinaryOp::kFastGeluBackwardWithDyX
    const T dtanh = alpha * (half * x + beta * static_cast<T>(1.5) * pow3);
    x_diff = (half + half * tanh_out + dtanh * (one - tanh_out * tanh_out)) * m * dy;
  }
};

template<>
struct FusedFastGeluMulGradFunctor<half> {
  static constexpr float alpha = FusedFastGeluMulGradFunctor<float>::alpha;
  static constexpr float beta = FusedFastGeluMulGradFunctor<float>::beta;
  FusedFastGeluMulGradFunctor<float> float_functor;

  __device__ FusedFastGeluMulGradFunctor() {}

  __device__ void operator()(half& x_diff, half& m_diff, const half& dy, const half& x,
                             const half& m) const {
#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
    const half halpha = __float2half_rn(alpha);
    const half hbeta = __float2half_rn(beta);
    const half hone = __float2half_rn(1.0F);
    const half hhalf = __float2half_rn(0.5F);
    const half pow3 = x * x * x;
    const float tanh_in = __half2float(halpha * (x + hbeta * pow3));
    const half tanh_out = __float2half_rn(TanhApprox(tanh_in));
    // m_diff
    m_diff = hhalf * x * (hone + tanh_out) * dy;
    // x_diff
    const half dtanh = halpha * (hhalf * x + hbeta * __float2half_rn(1.5F) * pow3);
    x_diff = (hhalf + hhalf * tanh_out + dtanh * (hone - tanh_out * tanh_out)) * m * dy;
#else
    float x_diff_float;
    float m_diff_float;
    float_functor(x_diff_float, m_diff_float, static_cast<float>(dy), static_cast<float>(x),
                  static_cast<float>(m));
    x_diff = static_cast<half>(x_diff_float);
    m_diff = static_cast<half>(m_diff_float);
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  }

#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  __device__ void Apply2(half* x_diff, half* m_diff, const half* dy, const half* x,
                         const half* m) const {
    const half2 dy2 = *(reinterpret_cast<const half2*>(dy));
    const half2 x2 = *(reinterpret_cast<const half2*>(x));
    const half2 m2 = *(reinterpret_cast<const half2*>(m));
    const half2 alpha2 = __float2half2_rn(alpha);
    const half2 beta2 = __float2half2_rn(beta);
    const half2 one2 = __float2half2_rn(1.0F);
    const half2 hhalf2 = __float2half2_rn(0.5F);
    const half2 pow3 = __hmul2(__hmul2(x2, x2), x2);
    const float2 tanh_in = __half22float2(__hmul2(alpha2, __hadd2(x2, __hmul2(beta2, pow3))));
    float2 tanh_out;
    tanh_out.x = TanhApprox(tanh_in.x);
    tanh_out.y = TanhApprox(tanh_in.y);
    const half2 tanh_out2 = __float22half2_rn(tanh_out);
    // m_diff
    const half2 m_diff2 = __hmul2(__hmul2(hhalf2, __hmul2(x2, __hadd2(one2, tanh_out2))), dy2);
    // x_diff
    const half2 dtanh = __hmul2(
        alpha2,
        __hadd2(__hmul2(hhalf2, x2), __hmul2(beta2, __hmul2(pow3, __float2half2_rn(1.5F)))));
    const half2 x_diff2 =
        __hmul2(__hmul2(__hadd2(__hadd2(hhalf2, __hmul2(hhalf2, tanh_out2)),
                                __hmul2(dtanh, __hsub2(one2, __hmul2(tanh_out2, tanh_out2)))),
                        m2),
                dy2);
    *reinterpret_cast<half2*>(x_diff) = x_diff2;
    *reinterpret_cast<half2*>(m_diff) = m_diff2;
  }
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
};

#if CUDA_VERSION >= 11000

template<>
struct FusedFastGeluMulGradFunctor<nv_bfloat16> {
  FusedFastGeluMulGradFunctor<float> float_functor;

  __device__ FusedFastGeluMulGradFunctor() {}

  __device__ void operator()(nv_bfloat16& x_diff, nv_bfloat16& m_diff, const nv_bfloat16& dy,
                             const nv_bfloat16& x, const nv_bfloat16& m) const {
    float x_diff_float;
    float m_diff_float;
    float_functor(x_diff_float, m_diff_float, __bfloat162float(dy), __bfloat162float(x),
                  __bfloat162float(m));
    x_diff = __float2bfloat16(x_diff_float);
    m_diff = __float2bfloat16(m_diff_float);
  }
};

#endif  // CUDA_VERSION >= 11000

template<int pack_size, typename FunctorT, typename T>
__device__ __forceinline__
    typename std::enable_if<elementwise::HasApply2<FunctorT>::value == true && pack_size % 2 == 0,
                            void>::type
    FusedFastGeluMulGradFunctorApplyPack(const FunctorT& functor,
                                         elementwise::Packed<T, pack_size>& x_diff_pack,
                                         elementwise::Packed<T, pack_size>& m_diff_pack,
                                         const elementwise::Packed<T, pack_size>& dy_pack,
                                         const elementwise::Packed<T, pack_size>& x_pack,
                                         const elementwise::Packed<T, pack_size>& m_pack) {
#pragma unroll
  for (int j = 0; j < pack_size; j += 2) {
    functor.Apply2(x_diff_pack.elem + j, m_diff_pack.elem + j, dy_pack.elem + j, x_pack.elem + j,
                   m_pack.elem + j);
  }
}

template<int pack_size, typename FunctorT, typename T>
__device__ __forceinline__
    typename std::enable_if<elementwise::HasApply2<FunctorT>::value == false || pack_size % 2 != 0,
                            void>::type
    FusedFastGeluMulGradFunctorApplyPack(const FunctorT& functor,
                                         elementwise::Packed<T, pack_size>& x_diff_pack,
                                         elementwise::Packed<T, pack_size>& m_diff_pack,
                                         const elementwise::Packed<T, pack_size>& dy_pack,
                                         const elementwise::Packed<T, pack_size>& x_pack,
                                         const elementwise::Packed<T, pack_size>& m_pack) {
#pragma unroll
  for (int j = 0; j < pack_size; ++j) {
    functor(x_diff_pack.elem[j], m_diff_pack.elem[j], dy_pack.elem[j], x_pack.elem[j],
            m_pack.elem[j]);
  }
}

template<int pack_size, typename T>
__global__ void __launch_bounds__(elementwise::kBlockSize)
    FusedFastGeluMulGradCudaKernel(int64_t n_pack, elementwise::Packed<T, pack_size>* x_diff_pack,
                                   elementwise::Packed<T, pack_size>* m_diff_pack,
                                   const elementwise::Packed<T, pack_size>* dy_pack,
                                   const elementwise::Packed<T, pack_size>* x_pack,
                                   const elementwise::Packed<T, pack_size>* m_pack, int64_t n_tail,
                                   T* x_diff_tail, T* m_diff_tail, const T* dy_tail,
                                   const T* x_tail, const T* m_tail) {
  FusedFastGeluMulGradFunctor<T> functor;
  const int global_tid = blockIdx.x * elementwise::kBlockSize + threadIdx.x;
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    FusedFastGeluMulGradFunctorApplyPack<pack_size>(functor, x_diff_pack[i], m_diff_pack[i],
                                                    dy_pack[i], x_pack[i], m_pack[i]);
  }
  if (global_tid < n_tail) {
    functor(x_diff_tail[global_tid], m_diff_tail[global_tid], dy_tail[global_tid],
            x_tail[global_tid], m_tail[global_tid]);
  }
}

template<size_t pack_size, typename T>
cudaError_t LaunchFusedFastGeluMulGradCudaKernelByPack(cudaStream_t stream, int64_t n, T* x_diff,
                                                       T* m_diff, const T* dy, const T* x,
                                                       const T* m) {
  const int64_t n_pack = n / pack_size;
  const int64_t tail_offset = n_pack * pack_size;
  const int64_t n_tail = n - tail_offset;
  int num_blocks;
  {
    cudaError_t err = elementwise::GetNumBlocks(n_pack, &num_blocks);
    if (err != cudaSuccess) { return err; }
  }
  FusedFastGeluMulGradCudaKernel<pack_size><<<num_blocks, elementwise::kBlockSize, 0, stream>>>(
      n_pack, reinterpret_cast<elementwise::Packed<T, pack_size>*>(x_diff),
      reinterpret_cast<elementwise::Packed<T, pack_size>*>(m_diff),
      reinterpret_cast<const elementwise::Packed<T, pack_size>*>(dy),
      reinterpret_cast<const elementwise::Packed<T, pack_size>*>(x),
      reinterpret_cast<const elementwise::Packed<T, pack_size>*>(m), n_tail, x_diff + tail_offset,
      m_diff + tail_offset, dy + tail_offset, x + tail_offset, m + tail_offset);
  return cudaPeekAtLastError();
}

template<typename T>
static cudaError_t LaunchFusedFastGeluMulGradCudaKernel(cudaStream_t stream, int64_t n, T* x_diff,
                                                        T* m_diff, const T* dy, const T* x,
                                                        const T* m) {
  constexpr int max_pack_size = elementwise::PackSize<T>();
  if (elementwise::IsAlignedForPack<max_pack_size>(x_diff, m_diff, dy, x, m)) {
    return LaunchFusedFastGeluMulGradCudaKernelByPack<max_pack_size>(stream, n, x_diff, m_diff, dy,
                                                                     x, m);
  } else {
    return LaunchFusedFastGeluMulGradCudaKernelByPack<1>(stream, n, x_diff, m_diff, dy, x, m);
  }
}

}  // namespace fused_gelu

template<typename T>
class FusedFastGeluMulGradKernel final : public user_op::OpKernel {
 public:
  FusedFastGeluMulGradKernel() = default;
  ~FusedFastGeluMulGradKernel() override = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* out_diff = ctx->Tensor4ArgNameAndIndex("out_diff", 0);
    const auto* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto* multiplier = ctx->Tensor4ArgNameAndIndex("multiplier", 0);
    auto* in_diff = ctx->Tensor4ArgNameAndIndex("in_diff", 0);
    auto* multiplier_diff = ctx->Tensor4ArgNameAndIndex("multiplier_diff", 0);

    int64_t elem_cnt = in->shape_view().elem_cnt();
    OF_CUDA_CHECK((fused_gelu::LaunchFusedFastGeluMulGradCudaKernel(
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), elem_cnt, in_diff->mut_dptr<T>(),
        multiplier_diff->mut_dptr<T>(), out_diff->dptr<T>(), in->dptr<T>(),
        multiplier->dptr<T>())));
  };
};

#define REGISTER_FUSED_FAST_GELU_MUL_GRAD_CUDA_KERNEL(dtype)           \
  REGISTER_USER_KERNEL("fused_fast_gelu_mul_grad")                     \
      .SetCreateFn<FusedFastGeluMulGradKernel<dtype>>()                \
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
