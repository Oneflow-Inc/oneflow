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
#include "oneflow/core/ep/common/primitive/elementwise_unary.h"
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

namespace cutlass {
namespace epilogue {
namespace thread {

template<typename ElementOutput_, int Count, typename ElementAccumulator_ = ElementOutput_,
         typename ElementCompute_ = ElementOutput_,
         FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class RightFastGeluAndMul {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  struct Params {};

 private:
  ElementCompute alpha_;
  ElementCompute beta_;

 public:
  CUTLASS_HOST_DEVICE
  RightFastGeluAndMul(Params const& /*params*/) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) { assert(false); }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& lhs, FragmentAccumulator const& rhs) const {
    NumericArrayConverter<ElementOutput, ElementAccumulator, kCount, Round> accumulator_to_output;

    FragmentOutput converted_lhs = accumulator_to_output(lhs);
    FragmentOutput converted_rhs = accumulator_to_output(rhs);

    cutlass::epilogue::thread::GELU_taylor<FragmentOutput> fast_gelu;
    cutlass::multiplies<FragmentOutput> mul;
    auto fast_gelu_rhs = fast_gelu(converted_rhs);
    return mul(fast_gelu_rhs, converted_lhs);
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(ElementAccumulator const& lhs, ElementAccumulator const& rhs) const {
    ElementOutput convert_lhs(lhs);
    ElementOutput convert_rhs(rhs);
    cutlass::epilogue::thread::GELU_taylor<ElementOutput> fast_gelu;
    cutlass::multiplies<ElementOutput> mul;
    auto fast_gelu_lhs = fast_gelu(convert_lhs);
    return mul(fast_gelu_lhs, convert_rhs);
  }
};
}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

namespace oneflow {

namespace {

template<typename T>
void DualGemmGeglu(ep::CudaStream* stream, int32_t m, int32_t n, int32_t k, const T* x, const T* w,
                   const T* b, T* y) {
  constexpr int kStages = 5;
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
  using EpilogueOutputOp2 = cutlass::epilogue::thread::RightFastGeluAndMul<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementOutput,
      ElementCompute>;

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

  int split_k_slices = DualGemm::kSplitKSerial ? 2 : 1;
  typename cutlass::TensorRef<typename DualGemm::ElementC, typename DualGemm::LayoutC>
      nullptr_ref{};

  typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::RowMajor> tensor_a0(
      reinterpret_cast<const cutlass::half_t*>(x), k);
  typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::ColumnMajor> tensor_b0(
      reinterpret_cast<const cutlass::half_t*>(w), k);
  typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::ColumnMajor> tensor_b1(
      reinterpret_cast<const cutlass::half_t*>(w + n * k), k);
  typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::RowMajor> tensor_bias0(
      reinterpret_cast<const cutlass::half_t*>(b), {0});
  typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::RowMajor> tensor_bias1(
      reinterpret_cast<const cutlass::half_t*>(b + n), {0});
  typename cutlass::TensorRef<ElementOperandA, cutlass::layout::RowMajor> tensor_out(
      reinterpret_cast<cutlass::half_t*>(y), n);

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  typename DualGemm::Arguments arguments{
      problem_size,    tensor_a0,    tensor_b0,     tensor_bias0, nullptr_ref,
      tensor_b1,       tensor_bias1, nullptr_ref,   tensor_out,   {alpha0, beta0},
      {alpha1, beta1}, {},           split_k_slices};

  DualGemm dual_gemm_op;
  dual_gemm_op.initialize(arguments, stream->cublas_workspace(), stream->cuda_stream());
  dual_gemm_op();
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
  ep::primitive::UnaryFunctor<DeviceType::kCUDA, ep::primitive::UnaryOp::kFastGelu, T, T> fast_gelu(
      0, 0);
  T gelu_gate = fast_gelu(gate);

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

    if (ParseBooleanFromEnv("ONEFLOW_KERNEL_EANBLE_DUAL_GEMM_GLU", false)) {
      DualGemmGeglu<half>(ctx->stream()->As<ep::CudaStream>(), num_samples, out_size, in_size,
                          in->dptr<half>(), w->dptr<half>(), b->dptr<half>(),
                          out->mut_dptr<half>());
      return;
    }

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
