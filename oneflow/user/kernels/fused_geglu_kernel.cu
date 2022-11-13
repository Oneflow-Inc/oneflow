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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/cuda/primitive/unary_functor.cuh"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

namespace oneflow {

namespace {

template<typename T>
void DualGemmGeglu(int32_t m, int32_t n, int32_t k, const T* x, const T* w, const T* b) {
  constexpr int kStages = 3;
  constexpr bool kSplitKSerial = false;
  constexpr bool kUseBias = true;
  using ElementOperandA = cutlass::half_t;
  using ElementOperandB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  constexpr auto kScaleType =
      kUseBias ? cutlass::epilogue::thread::ScaleType::NoBetaScaling
               : (
                   // No bias
                   kSplitKSerial ? cutlass::epilogue::thread::ScaleType::Default
                                 : cutlass::epilogue::thread::ScaleType::Nothing);
  using EpilogueOutputOp0 =
      cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                   128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                   ElementAccumulator, ElementCompute, kScaleType>;
  using EpilogueOutputOp1 =
      cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                   128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                   ElementAccumulator, ElementCompute, kScaleType>;
  using EpilogueOutputOp2 =
      cutlass::epilogue::thread::LeftSiLUAndMul<ElementOutput,
                                                128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                ElementOutput, ElementCompute>;

  const ElementCompute alpha0 = ElementCompute(1);
  const ElementCompute beta0 = ElementCompute(kUseBias ? 1 : 0);
  const ElementCompute alpha1 = ElementCompute(1);
  const ElementCompute beta1 = ElementCompute(kUseBias ? 1 : 0);

  // Optionally, we might not need intermediate GEMM outputs
  constexpr bool kStoreD0 = false;
  constexpr bool kStoreD1 = false;
  using DualGemm = cutlass::gemm::device::DualGemm<
      ElementOperandA, cutlass::layout::RowMajor, ElementOperandB, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp0,
      EpilogueOutputOp1, EpilogueOutputOp2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, kStages, kStoreD0, kStoreD1,
      kSplitKSerial>;

  int split_k_slices = Gemm0::kSplitKSerial ? 2 : 1;

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  typename DualGemm::Arguments arguments{problem_size,
                                         tensor_A0.device_ref(),
                                         tensor_B0.device_ref(),
                                         ref_B0,
                                         DualGemm::kStoreD0 ? tensor_D0.device_ref() : nullptr_ref,
                                         tensor_B1.device_ref(),
                                         ref_B1,
                                         DualGemm::kStoreD1 ? tensor_D1.device_ref() : nullptr_ref,
                                         tensor_D2.device_ref(),
                                         {alpha0, beta0},
                                         {alpha1, beta1},
                                         {},
                                         split_k_slices};
}

template<typename T>
__device__ T Gelu(T x) {
  return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + erf(static_cast<T>(M_SQRT1_2) * x));
}

/*! Noted
 * mixture precision isn't allowed
 * hence we need to explicitly cast half or nv_bfloat16 type to float
 * and after finishing the gelu calculation, explicitly cast back
 */
template<>
__device__ half Gelu(half x) {
  return static_cast<half>(Gelu<float>(static_cast<float>(x)));
}

#if CUDA_VERSION >= 11000
template<>
__device__ nv_bfloat16 Gelu(nv_bfloat16 x) {
  return static_cast<nv_bfloat16>(Gelu<float>(static_cast<float>(x)));
}
#endif  // CUDA_VERSION >= 11000

template<typename T>
__global__ void FusedGegluForwardGpu(const int in_size, const int out_size, const int num_sample,
                                     const T* matmul, const T* b, T* y) {
  // obtain the index of current thread
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size * num_sample) { return; }

  // obtain index of row and col in the output tensor
  const int64_t out_row = i / out_size;
  const int64_t out_col = i - out_row * out_size;

  // obtain col index in both matmul tensor and bias tensor
  const int64_t x1_col = out_col;
  const int64_t x2_col = out_col + out_size;

  // obtain element before gelu and element-wise product
  T hidden_state = matmul[2 * out_row * out_size + x1_col] + b[x1_col];
  T gate = matmul[2 * out_row * out_size + x2_col] + b[x2_col];

  // calculate gelu
  T gelu_gate = Gelu<T>(gate);

  // calculate element-wise product
  y[i] = gelu_gate * hidden_state;
}

struct alignas(8) Half4 {
  half2 x;
  half2 y;
};

__device__ Half4 Hmul4(const Half4& a, const Half4& b) {
  Half4 r;
  r.x = __hmul2(a.x, b.x);
  r.y = __hmul2(a.y, b.y);
  return r;
}

__device__ Half4 Hadd4(const Half4& a, const Half4& b) {
  Half4 r;
  r.x = __hadd2(a.x, b.x);
  r.y = __hadd2(a.y, b.y);
  return r;
}

__global__ void FusedGegluHalf4ForwardGpu(const int in_size, const int out_size,
                                          const int num_sample, const Half4* matmul, const Half4* b,
                                          Half4* y) {
  // obtain the index of current thread
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size * num_sample) { return; }

  // obtain index of row and col in the output tensor
  const int32_t out_row = i / out_size;
  const int32_t out_col = i - out_row * out_size;

  // obtain col index in both matmul tensor and bias tensor
  const int32_t x1_col = out_col;
  const int32_t x2_col = out_col + out_size;

  // obtain element before gelu and element-wise product
  const Half4 hidden_state = Hadd4(matmul[2 * out_row * out_size + x1_col], b[x1_col]);
  const Half4 gate = Hadd4(matmul[2 * out_row * out_size + x2_col], b[x2_col]);

  // calculate gelu
  ep::primitive::UnaryFunctor<DeviceType::kCUDA, ep::primitive::UnaryOp::kFastGelu, half, half>
      fast_gelu(0, 0);
  Half4 gelu_out;

#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
  fast_gelu.Apply2(reinterpret_cast<half*>(&gelu_out.x), reinterpret_cast<const half*>(&gate.x));
  fast_gelu.Apply2(reinterpret_cast<half*>(&gelu_out.y), reinterpret_cast<const half*>(&gate.y));
#endif

  // calculate element-wise product
  y[i] = Hmul4(gelu_out, hidden_state);
}

}  // namespace

template<typename T>

class GpuFusedGegluKernel final : public user_op::OpKernel {
 public:
  GpuFusedGegluKernel() = default;
  ~GpuFusedGegluKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    // obtain corresponding tensors from the context
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* w = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("bias", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* matmul_out = ctx->Tensor4ArgNameAndIndex("matmul_out", 0);

    // obtain dimensions
    const int64_t in_size = in->shape_view().At(1);
    const int64_t out_size = out->shape_view().At(1);
    const int64_t num_samples = in->shape_view().At(0);

    // calculate X*W through cuBLAS
    // ref -> reduce_kernel.cpp -> matmul
    auto matmul = ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
        DeviceType::kCUDA, in->data_type(), ep::primitive::BlasTransposeType::N,
        ep::primitive::BlasTransposeType::T);
    CHECK(matmul);
    /* Launch(Stream* stream, size_t m, size_t n, size_t k, Scalar alpha, const void* a,
                  const void* b, Scalar beta, void* c) = 0; */
    matmul->Launch(ctx->stream(), num_samples, out_size * 2, in_size, 1.0, in->dptr(), w->dptr(),
                   0.0, matmul_out->mut_dptr());

    // invoke fused geglu kernel
    if (out_size % 4 == 0 && in->data_type() == DataType::kFloat16) {
      int n = out_size * num_samples / 4;
      FusedGegluHalf4ForwardGpu<<<(n + 256 - 1) / 256, 256, 0,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          in_size / 4,                                          /* in_size */
          out_size / 4,                                         /* out_size */
          num_samples,                                          /* element_cnt */
          reinterpret_cast<const Half4*>(matmul_out->dptr<>()), /* matmul result */
          reinterpret_cast<const Half4*>(b->dptr()),            /* bias */
          reinterpret_cast<Half4*>(out->mut_dptr()) /* output tensor */);

    } else {
      int n = out_size * num_samples;
      FusedGegluForwardGpu<T>
          <<<(n + 256 - 1) / 256, 256, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              in_size,               /* in_size */
              out_size,              /* out_size */
              num_samples,           /* element_cnt */
              matmul_out->dptr<T>(), /* matmul result */
              b->dptr<T>(),          /* bias */
              out->mut_dptr<T>() /* output tensor */);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_FUSED_GEGLU_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("fused_geglu")                                  \
      .SetCreateFn<GpuFusedGegluKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_FUSED_GEGLU_KERNEL(float)
REGISTER_GPU_FUSED_GEGLU_KERNEL(double)
REGISTER_GPU_FUSED_GEGLU_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_GPU_FUSED_GEGLU_KERNEL(nv_bfloat16)
#endif  // CUDA_VERSION >= 11000

}  // namespace oneflow
