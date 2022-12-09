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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/ep/cuda/primitive/unary_functor.cuh"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

#if !defined(__clang__)

#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

namespace cutlass {
namespace epilogue {
namespace thread {

template<typename ElementOutput_, int Count, template<typename> typename Activation,
         typename ElementAccumulator_ = ElementOutput_, typename ElementCompute_ = ElementOutput_,
         FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class RightActivationAndMul {
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
  RightActivationAndMul(Params const& /*params*/) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) { assert(false); }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& lhs, FragmentAccumulator const& rhs) const {
    NumericArrayConverter<ElementOutput, ElementAccumulator, kCount, Round> accumulator_to_output;

    FragmentOutput converted_lhs = accumulator_to_output(lhs);
    FragmentOutput converted_rhs = accumulator_to_output(rhs);

    Activation<FragmentOutput> act;
    cutlass::multiplies<FragmentOutput> mul;
    auto act_rhs = act(converted_rhs);
    return mul(act_rhs, converted_lhs);
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(ElementAccumulator const& lhs, ElementAccumulator const& rhs) const {
    ElementOutput convert_lhs(lhs);
    ElementOutput convert_rhs(rhs);
    Activation<ElementOutput> act;
    cutlass::multiplies<ElementOutput> mul;
    auto act_lhs = fast_gelu(convert_lhs);
    return mul(act_lhs, convert_rhs);
  }
};
}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

#endif  // !defined(__clang__)

namespace oneflow {

namespace {

#if !defined(__clang__)

template<typename T>
struct GetCutlassType {
  using type = T;
};

template<>
struct GetCutlassType<half> {
  using type = cutlass::half_t;
};

#if CUDA_VERSION >= 11000

template<>
struct GetCutlassType<nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

#endif

template<typename Acc, typename Arch, template<typename> typename Activation>
void DualGemmGegluHalf(ep::CudaStream* stream, int32_t m, int32_t n, int32_t k, const void* x,
                       const void* w, const void* v, const void* b, const void* c, void* wx,
                       int32_t wx_stride, void* vx, int32_t vx_stride, void* y) {
  constexpr int kStages = 5;
  constexpr bool kSplitKSerial = false;
  constexpr bool kUseBias = true;
  using ElementOperandA = cutlass::half_t;
  using ElementOperandB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = Acc;
  using ElementCompute = Acc;
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
  using EpilogueOutputOp2 = cutlass::epilogue::thread::RightActivationAndMul<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, Activation, ElementOutput,
      ElementCompute>;

  const ElementCompute alpha0 = ElementCompute(1);
  const ElementCompute beta0 = ElementCompute(kUseBias ? 1 : 0);
  const ElementCompute alpha1 = ElementCompute(1);
  const ElementCompute beta1 = ElementCompute(kUseBias ? 1 : 0);

  // Optionally, we might not need intermediate GEMM outputs
  constexpr bool kStoreD0 = true;
  constexpr bool kStoreD1 = true;
  using DualGemm = cutlass::gemm::device::DualGemm<
      ElementOperandA, cutlass::layout::RowMajor, ElementOperandB, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
      Arch, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp0, EpilogueOutputOp1,
      EpilogueOutputOp2, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, kStages,
      kStoreD0, kStoreD1, kSplitKSerial>;

  int split_k_slices = DualGemm::kSplitKSerial ? 2 : 1;

  typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::RowMajor> tensor_a0(
      reinterpret_cast<const cutlass::half_t*>(x), k);
  typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::ColumnMajor> tensor_b0(
      reinterpret_cast<const cutlass::half_t*>(w), k);
  typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::ColumnMajor> tensor_b1(
      reinterpret_cast<const cutlass::half_t*>(v), k);
  typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::RowMajor> tensor_bias0(
      reinterpret_cast<const cutlass::half_t*>(b), {0});
  typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::RowMajor> tensor_bias1(
      reinterpret_cast<const cutlass::half_t*>(c), {0});
  typename cutlass::TensorRef<typename DualGemm::ElementC, typename DualGemm::LayoutC> tensor_d0(
      reinterpret_cast<cutlass::half_t*>(wx), wx_stride);
  typename cutlass::TensorRef<typename DualGemm::ElementC, typename DualGemm::LayoutC> tensor_d1(
      reinterpret_cast<cutlass::half_t*>(vx), vx_stride);
  typename cutlass::TensorRef<ElementOperandA, cutlass::layout::RowMajor> tensor_out(
      reinterpret_cast<cutlass::half_t*>(y), n);

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  typename DualGemm::Arguments arguments{
      problem_size,    tensor_a0,    tensor_b0,     tensor_bias0, tensor_d0,
      tensor_b1,       tensor_bias1, tensor_d1,     tensor_out,   {alpha0, beta0},
      {alpha1, beta1}, {},           split_k_slices};

  DualGemm dual_gemm_op;
  dual_gemm_op.initialize(arguments, stream->cublas_workspace(), stream->cuda_stream());
  dual_gemm_op(stream->cuda_stream());
}

template<typename Acc, typename Arch>
bool TryDispatchDualGemmImplActivation(ep::CudaStream* stream, const std::string& activation,
                                       int32_t m, int32_t n, int32_t k, const void* x,
                                       const void* w, const void* v, const void* b, const void* c,
                                       void* wx, int32_t wx_stride, void* vx, int32_t vx_stride,
                                       void* y) {
  if (activation == "fast_gelu") {
    DualGemmGegluHalf<Acc, Arch, cutlass::epilogue::thread::GELU_taylor>(
        stream, m, n, k, x, w, v, b, c, wx, wx_stride, vx, vx_stride, y);
    return true;
  } else if (activation == "gelu") {
    DualGemmGegluHalf<Acc, Arch, cutlass::epilogue::thread::GELU>(stream, m, n, k, x, w, v, b, c,
                                                                  wx, wx_stride, vx, vx_stride, y);
    return true;
  } else {
    return false;
  }
}

template<typename T, typename Arch>
bool TryDispatchDualGemmImplAccType(ep::CudaStream* stream, const std::string& activation,
                                    int32_t m, int32_t n, int32_t k, const T* x, const T* w,
                                    const T* v, const T* b, const T* c, T* wx, int32_t wx_stride,
                                    T* vx, int32_t vx_stride, T* y) {
  const static bool allow_half_precision =
      ParseBooleanFromEnv("ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION", false);
  if (std::is_same<T, half>::value) {
    if (allow_half_precision) {
      return TryDispatchDualGemmImplActivation<cutlass::half_t, Arch>(
          stream, activation, m, n, k, x, w, v, b, c, wx, wx_stride, vx, vx_stride, y);
    } else {
      return TryDispatchDualGemmImplActivation<float, Arch>(stream, activation, m, n, k, x, w, v, b,
                                                            c, wx, wx_stride, vx, vx_stride, y);
    }
  } else {
    return false;
  }
}

template<typename T, typename Arch>
bool TryDispatchDualGemmImplAlignment(ep::CudaStream* stream, const std::string& activation,
                                      int32_t m, int32_t n, int32_t k, const T* x, const T* w,
                                      const T* v, const T* b, const T* c, T* wx, int32_t wx_stride,
                                      T* vx, int32_t vx_stride, T* y) {
  if (m % 8 == 0 && n % 8 == 0 && k % 8 == 0
      && reinterpret_cast<uintptr_t>(x) % (8 * sizeof(T)) == 0
      && reinterpret_cast<uintptr_t>(w) % (8 * sizeof(T)) == 0
      && reinterpret_cast<uintptr_t>(v) % (8 * sizeof(T)) == 0
      && reinterpret_cast<uintptr_t>(b) % (8 * sizeof(T)) == 0
      && reinterpret_cast<uintptr_t>(c) % (8 * sizeof(T)) == 0
      && reinterpret_cast<uintptr_t>(wx) % (8 * sizeof(T)) == 0 && wx_stride % 8 == 0
      && reinterpret_cast<uintptr_t>(vx) % (8 * sizeof(T)) == 0
      && reinterpret_cast<uintptr_t>(y) % (8 * sizeof(T)) == 0 && vx_stride % 8 == 0) {
    return TryDispatchDualGemmImplAccType<T, Arch>(stream, activation, m, n, k, x, w, v, b, c, wx,
                                                   wx_stride, vx, vx_stride, y);
  } else {
    return false;
  }
}

template<typename T>
bool TryDispatchDualGemmImplArchTag(ep::CudaStream* stream, const std::string& activation,
                                    int32_t m, int32_t n, int32_t k, const T* x, const T* w,
                                    const T* v, const T* b, const T* c, T* wx, int32_t wx_stride,
                                    T* vx, int32_t vx_stride, T* y) {
  const int arch = stream->cuda_arch();
  if (arch == 800) {
    return TryDispatchDualGemmImplAlignment<T, cutlass::arch ::Sm80>(
        stream, activation, m, n, k, x, w, v, b, c, wx, wx_stride, vx, vx_stride, y);
  } else {
    return false;
  }
}

#endif  // !defined(__clang__)
template<typename T>
bool TryDispatchDualGemmImpl(ep::CudaStream* stream, const std::string& activation, int32_t m,
                             int32_t n, int32_t k, const T* x, const T* w, const T* v, const T* b,
                             const T* c, T* wx, int32_t wx_stride, T* vx, int32_t vx_stride, T* y) {
#if !defined(__clang__)
  const static bool enabled =
      ParseBooleanFromEnv("ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL", false);
  if (enabled) {
    return TryDispatchDualGemmImplArchTag<T>(stream, activation, m, n, k, x, w, v, b, c, wx,
                                             wx_stride, vx, vx_stride, y);
  } else {
    return false;
  }
#else
  return false;
#endif  // !defined(__clang__)
}

template<typename T, typename IndexType, ep::primitive::UnaryOp act_type, int32_t pack_size>
__global__ void FusedGluForwardGpu(
    const IndexType m, const IndexType packed_n, const IndexType k, const IndexType packed_stride,
    const IndexType packed_num, ep::primitive::UnaryFunctor<DeviceType::kCUDA, act_type, T, T> act,
    const T* matmul_wx, const T* b, const T* matmul_vx, const T* c, T* y) {
  // obtain global thread index
  IndexType global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  // define type of Pack
  using LoadPack = cuda::elementwise::Packed<T, pack_size>;

  // workload of current thread
  for (IndexType packed_index = global_thread_id, step = gridDim.x * blockDim.x;
       packed_index < packed_num; packed_index += step) {
    // obtain the row and col index in output tensor "y"
    const IndexType y_packed_row = packed_index / packed_n;
    const IndexType y_packed_col = packed_index - y_packed_row * packed_n;

    // cast type to load type
    const LoadPack* matmul_wx_load = reinterpret_cast<const LoadPack*>(matmul_wx)
                                     + (y_packed_row * packed_stride + y_packed_col);
    const LoadPack* matmul_vx_load = reinterpret_cast<const LoadPack*>(matmul_vx)
                                     + (y_packed_row * packed_stride + y_packed_col);
    const LoadPack* b_load = reinterpret_cast<const LoadPack*>(b) + y_packed_col;
    const LoadPack* c_load = reinterpret_cast<const LoadPack*>(c) + y_packed_col;

    // init vectors
    LoadPack matmul_wx_vec = *matmul_wx_load;
    LoadPack matmul_vx_vec = *matmul_vx_load;
    LoadPack b_vec = *b_load;
    LoadPack c_vec = *c_load;
    LoadPack y_vec;

#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      // calculate the hidden_state and gate
      T hidden_state = matmul_wx_vec.elem[i] + b_vec.elem[i];
      T gate = matmul_vx_vec.elem[i] + c_vec.elem[i];

      // calculate activation
      T act_gate = act(gate);

      // calculate element-wise product
      y_vec.elem[i] = hidden_state * act_gate;
    }
    *(reinterpret_cast<LoadPack*>(y + packed_index * pack_size)) = y_vec;
  }
}

template<typename T, typename IndexType, ep::primitive::UnaryOp act_type, int32_t pack_size>
void LaunchFusedGluForwardGpu(ep::Stream* stream, const int64_t m, const int64_t packed_n,
                              const int64_t k, int64_t packed_stride, const T* matmul_wx,
                              const T* b, const T* matmul_vx, const T* c, T* y) {
  ep::primitive::UnaryFunctor<DeviceType::kCUDA, act_type, T, T> act(0, 0);
  int64_t pack_num = m * packed_n;
  constexpr int32_t block_size = 128;
  unsigned int grid_size = (pack_num + block_size - 1) / block_size;
  FusedGluForwardGpu<T, IndexType, act_type, pack_size>
      <<<grid_size, block_size, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          m, packed_n, k, packed_stride, pack_num, act, matmul_wx, b, matmul_vx, c, y);
}

template<typename T, ep::primitive::UnaryOp act_type, int32_t pack_size>
void DispatchIndexType(ep::Stream* stream, const int64_t m, const int64_t n, const int64_t k,
                       int64_t packed_stride, const T* matmul_wx, const T* b, const T* matmul_vx,
                       const T* c, T* y) {
  // convert n based on pack size
  const int64_t packed_n = n / pack_size;

  // dispatch index type
  if (m * packed_n < (1 << 30)) {
    LaunchFusedGluForwardGpu<T, int32_t, act_type, pack_size>(stream, m, packed_n, k, packed_stride,
                                                              matmul_wx, b, matmul_vx, c, y);
  } else {
    LaunchFusedGluForwardGpu<T, int64_t, act_type, pack_size>(stream, m, packed_n, k, packed_stride,
                                                              matmul_wx, b, matmul_vx, c, y);
  }
}

template<typename T, ep::primitive::UnaryOp act_type>
void DispatchAlignment(ep::Stream* stream, const int64_t m, const int64_t n, const int64_t k,
                       int64_t stride, const T* matmul_wx, const T* b, const T* matmul_vx,
                       const T* c, T* y) {
  const auto IsAligned = [&](const size_t alignment) {
    const uintptr_t matmul_wx_ptr = reinterpret_cast<uintptr_t>(matmul_wx);
    const uintptr_t matmul_vx_ptr = reinterpret_cast<uintptr_t>(matmul_vx);
    const uintptr_t b_ptr = reinterpret_cast<uintptr_t>(b);
    const uintptr_t c_ptr = reinterpret_cast<uintptr_t>(c);
    const uintptr_t y_ptr = reinterpret_cast<uintptr_t>(y);

    return (/* memory address alignment */
            matmul_wx_ptr % alignment == 0 && matmul_vx_ptr % alignment == 0
            && b_ptr % alignment == 0 && c_ptr % alignment == 0
            && y_ptr % alignment == 0
            /* #element per row alignment */
            && n % (alignment / sizeof(T)) == 0);
  };

  if (IsAligned(16)) {
    switch (sizeof(T)) {
      case 8:
        DispatchIndexType<T, act_type, 2>(stream, m, n, k, stride / 2, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
      case 4:
        DispatchIndexType<T, act_type, 4>(stream, m, n, k, stride / 4, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
      case 2:
        DispatchIndexType<T, act_type, 8>(stream, m, n, k, stride / 8, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
      case 1:
        DispatchIndexType<T, act_type, 16>(stream, m, n, k, stride / 16, matmul_wx, b, matmul_vx, c,
                                           y);
        break;
      default:
        DispatchIndexType<T, act_type, 1>(stream, m, n, k, stride / 1, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
    }
  } else if (IsAligned(8)) {
    switch (sizeof(T)) {
      case 4:
        DispatchIndexType<T, act_type, 2>(stream, m, n, k, stride / 2, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
      case 2:
        DispatchIndexType<T, act_type, 4>(stream, m, n, k, stride / 4, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
      case 1:
        DispatchIndexType<T, act_type, 8>(stream, m, n, k, stride / 8, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
      default:
        DispatchIndexType<T, act_type, 1>(stream, m, n, k, stride / 1, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
    }
  } else if (IsAligned(4)) {
    switch (sizeof(T)) {
      case 2:
        DispatchIndexType<T, act_type, 2>(stream, m, n, k, stride / 2, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
      case 1:
        DispatchIndexType<T, act_type, 4>(stream, m, n, k, stride / 4, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
      default:
        DispatchIndexType<T, act_type, 1>(stream, m, n, k, stride / 1, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
    }
  } else if (IsAligned(2)) {
    switch (sizeof(T)) {
      case 1:
        DispatchIndexType<T, act_type, 2>(stream, m, n, k, stride / 2, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
      default:
        DispatchIndexType<T, act_type, 1>(stream, m, n, k, stride / 1, matmul_wx, b, matmul_vx, c,
                                          y);
        break;
    }
  } else {
    DispatchIndexType<T, act_type, 1>(stream, m, n, k, stride / 1, matmul_wx, b, matmul_vx, c, y);
  }
}

template<typename T>
void DispatchActivationType(ep::Stream* stream, const int64_t m, const int64_t n, const int64_t k,
                            int64_t stride, const T* matmul_wx, const T* b, const T* matmul_vx,
                            const T* c, T* y, const std::string& activation) {
  if (activation == "none") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kIdentity>(stream, m, n, k, stride, matmul_wx, b,
                                                            matmul_vx, c, y);
  } else if (activation == "sigmoid") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kSigmoid>(stream, m, n, k, stride, matmul_wx, b,
                                                           matmul_vx, c, y);
  } else if (activation == "relu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kRelu>(stream, m, n, k, stride, matmul_wx, b,
                                                        matmul_vx, c, y);
  } else if (activation == "gelu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kGelu>(stream, m, n, k, stride, matmul_wx, b,
                                                        matmul_vx, c, y);
  } else if (activation == "fast_gelu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kFastGelu>(stream, m, n, k, stride, matmul_wx, b,
                                                            matmul_vx, c, y);
  } else if (activation == "silu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kSilu>(stream, m, n, k, stride, matmul_wx, b,
                                                        matmul_vx, c, y);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
class GpuFusedGluKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  GpuFusedGluKernel() = default;
  ~GpuFusedGluKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // obtain tensors from context
    const user_op::Tensor* input_tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* input_tensor_w = ctx->Tensor4ArgNameAndIndex("w", 0);
    const user_op::Tensor* input_tensor_b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out_tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out_tensor_matmul_wx = ctx->Tensor4ArgNameAndIndex("matmul_wx", 0);

    // obtain optional tensors from context
    bool is_split_mode = false;
    user_op::Tensor* input_tensor_v = nullptr;
    user_op::Tensor* input_tensor_c = nullptr;
    user_op::Tensor* out_tensor_matmul_vx = nullptr;

    if (ctx->has_input("v", 0) && ctx->has_input("c", 0)) {
      input_tensor_v = ctx->Tensor4ArgNameAndIndex("v", 0);
      input_tensor_c = ctx->Tensor4ArgNameAndIndex("c", 0);
      out_tensor_matmul_vx = ctx->Tensor4ArgNameAndIndex("matmul_vx", 0);
      is_split_mode = true;
    } else {
      CHECK((!ctx->has_input("v", 0)) && (!(ctx->has_input("c", 0))));
    }

    // obtain tensor shapes
    const ShapeView& x_shape = input_tensor_x->shape_view();
    const ShapeView& w_shape = input_tensor_w->shape_view();
    const ShapeView& b_shape = input_tensor_b->shape_view();
    const ShapeView& y_shape = out_tensor_y->shape_view();

    // validate dimension and number of axes
    CHECK_GE(x_shape.NumAxes(), 2)
        << "number of axes of \'x\' should have be greater than 1, yet get " << x_shape.NumAxes();
    CHECK_EQ(w_shape.NumAxes(), 2)
        << "number of axes of \'w\' should have be equal to 2, yet get " << w_shape.NumAxes();
    CHECK_EQ(b_shape.NumAxes(), 1)
        << "number of axes of \'b\' should have be equal to 1, yet get " << b_shape.NumAxes();

    // check input tensor shapes
    size_t x_num_axes = x_shape.NumAxes();
    CHECK_EQ(w_shape.At(1), x_shape.At(x_num_axes - 1))
        << "dimension 1 of \'w\'(" << w_shape.At(1)
        << ") is not consistant with the last dimension of \'x\'(" << x_shape.At(x_num_axes - 1)
        << ")";
    CHECK_EQ(b_shape.At(0), w_shape.At(0))
        << "dimension 0 of \'b\'(" << b_shape.At(0)
        << ") is not consistant with dimension 0 of \'w\'(" << w_shape.At(0) << ")";
    if (!is_split_mode) {
      CHECK_EQ(w_shape.At(1) % 2, 0) << "dimension 1 of \'w\' is not divisible by 2";
    }

    // check optional input tensor shapes
    if (is_split_mode) {
      const ShapeView& v_shape = input_tensor_v->shape_view();
      const ShapeView& c_shape = input_tensor_b->shape_view();

      CHECK_EQ(v_shape.NumAxes(), 2)
          << "number of axes of \'v\' should have be equal to 2, yet get " << v_shape.NumAxes();
      CHECK_EQ(c_shape.NumAxes(), 1)
          << "number of axes of \'c\' should have be equal to 1, yet get " << c_shape.NumAxes();
      CHECK_EQ(v_shape, w_shape) << "the shape of \'v\' is not consistant with \'w\'";
      CHECK_EQ(c_shape, b_shape) << "the shape of \'c\' is not consistant with \'b\'";
    }

    // infer m, n, k
    const int64_t m = x_shape.Count(0, x_num_axes - 1);
    const int64_t n = y_shape.At(x_num_axes - 1);
    const int64_t k = x_shape.At(x_num_axes - 1);

    if (TryDispatchDualGemmImpl(
            ctx->stream()->As<ep::CudaStream>(), ctx->Attr<std::string>("activation"), m, n, k,
            input_tensor_x->dptr<T>(), input_tensor_w->dptr<T>(),
            is_split_mode ? input_tensor_v->dptr<T>() : input_tensor_w->dptr<T>() + n * k,
            input_tensor_b->dptr<T>(),
            is_split_mode ? input_tensor_c->dptr<T>() : input_tensor_b->dptr<T>() + n,
            out_tensor_matmul_wx->mut_dptr<T>(), is_split_mode ? n : 2 * n,
            is_split_mode ? out_tensor_matmul_vx->mut_dptr<T>()
                          : out_tensor_matmul_wx->mut_dptr<T>() + n,
            is_split_mode ? n : 2 * n, out_tensor_y->mut_dptr<T>())) {
      return;
    }

    // calculate matmul_wx (and matmul_vx) through primitive
    auto matmul = ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
        DeviceType::kCUDA, input_tensor_x->data_type(), ep::primitive::BlasTransposeType::N,
        ep::primitive::BlasTransposeType::T);
    CHECK(matmul);
    /* Launch(Stream* stream, size_t m, size_t n, size_t k, Scalar alpha, const void* a,
                  const void* b, Scalar beta, void* c) = 0; */
    if (is_split_mode) {
      matmul->Launch(ctx->stream(), m, n, k, 1.0, input_tensor_x->dptr(), input_tensor_w->dptr(),
                     0.0, out_tensor_matmul_wx->mut_dptr());
      matmul->Launch(ctx->stream(), m, n, k, 1.0, input_tensor_x->dptr(), input_tensor_v->dptr(),
                     0.0, out_tensor_matmul_vx->mut_dptr());
    } else {
      matmul->Launch(ctx->stream(), m, n * 2, k, 1.0, input_tensor_x->dptr(),
                     input_tensor_w->dptr(), 0.0, out_tensor_matmul_wx->mut_dptr());
    }

    // dispatch according to activation type
    DispatchActivationType<T>(
        ctx->stream(),
        /*m, n, k=*/m, n, k,
        /*stride=*/is_split_mode ? n : 2 * n,
        /*matmul_wx=*/out_tensor_matmul_wx->dptr<T>(),
        /*b=*/input_tensor_b->dptr<T>(),
        /*matmul_vx=*/
        is_split_mode ? out_tensor_matmul_vx->dptr<T>() : out_tensor_matmul_wx->dptr<T>() + n,
        /*c=*/is_split_mode ? input_tensor_c->dptr<T>() : input_tensor_b->dptr<T>() + n,
        /*y=*/out_tensor_y->mut_dptr<T>(),
        /*activation=*/ctx->Attr<std::string>("activation"));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_GPU_FUSED_GLU_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("fused_glu")                                    \
      .SetCreateFn<GpuFusedGluKernel<dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_FUSED_GLU_KERNEL(double)
REGISTER_GPU_FUSED_GLU_KERNEL(float)
REGISTER_GPU_FUSED_GLU_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_GPU_FUSED_GLU_KERNEL(nv_bfloat16)
#endif

}  // namespace oneflow
