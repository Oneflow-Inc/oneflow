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
#include "oneflow/user/kernels/cublas_fused_mlp_util.cuh"

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

#if CUDA_VERSION >= 11020

#ifdef WITH_CUTLASS

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
    auto act_rhs = act(convert_rhs);
    return mul(act_rhs, convert_lhs);
  }
};
}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

#endif  // WITH_CUTLASS

namespace oneflow {

namespace {

#ifdef WITH_CUTLASS

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
  const bool allow_half_precision =
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

#endif  // WITH_CUTLASS
template<typename T>
bool TryDispatchDualGemmImpl(ep::CudaStream* stream, const std::string& activation, int32_t m,
                             int32_t n, int32_t k, const T* x, const T* w, const T* v, const T* b,
                             const T* c, T* wx, int32_t wx_stride, T* vx, int32_t vx_stride, T* y) {
#ifdef WITH_CUTLASS
  const bool enabled = ParseBooleanFromEnv("ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL", true);
  if (enabled) {
    return TryDispatchDualGemmImplArchTag<T>(stream, activation, m, n, k, x, w, v, b, c, wx,
                                             wx_stride, vx, vx_stride, y);
  } else {
    return false;
  }
#else
  return false;
#endif  // WITH_CUTLASS
}

template<typename T, typename IndexType, ep::primitive::UnaryOp act_type, int32_t pack_size>
__global__ void FusedGluForwardGpu(
    const IndexType m, const IndexType packed_n, const IndexType packed_num,
    const IndexType packed_stride,
    ep::primitive::UnaryFunctor<DeviceType::kCUDA, act_type, T, T> act, T* matmul_wx, T* matmul_vx,
    T* y) {
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
    const LoadPack* matmul_wx_load =
        reinterpret_cast<LoadPack*>(matmul_wx) + (y_packed_row * packed_stride + y_packed_col);
    const LoadPack* matmul_vx_load =
        reinterpret_cast<LoadPack*>(matmul_vx) + (y_packed_row * packed_stride + y_packed_col);

    // init vectors
    LoadPack matmul_wx_vec = *matmul_wx_load;
    LoadPack matmul_vx_vec = *matmul_vx_load;
    LoadPack y_vec;

#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      // obtain the hidden_state and gate
      T hidden_state = matmul_wx_vec.elem[i];
      T gate = matmul_vx_vec.elem[i];

      // calculate activation
      T act_gate = act(gate);

      // calculate element-wise product
      y_vec.elem[i] = hidden_state * act_gate;
    }
    *(reinterpret_cast<LoadPack*>(y + packed_index * pack_size)) = y_vec;
  }
}

template<typename T, typename IndexType, ep::primitive::UnaryOp act_type, int32_t pack_size>
void LaunchFusedGluForwardGpu(ep::Stream* stream, const IndexType m, const IndexType packed_n,
                              const IndexType pack_num, const IndexType packed_stride, T* matmul_wx,
                              T* matmul_vx, T* y) {
  constexpr int32_t block_size = 128;
  unsigned int grid_size = (pack_num + block_size - 1) / block_size;
  ep::primitive::UnaryFunctor<DeviceType::kCUDA, act_type, T, T> act(0, 0);
  FusedGluForwardGpu<T, IndexType, act_type, pack_size>
      <<<grid_size, block_size, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          m, packed_n, pack_num, packed_stride, act, matmul_wx, matmul_vx, y);
}

template<typename T, ep::primitive::UnaryOp act_type, int32_t pack_size>
void DispatchIndexType(ep::Stream* stream, const int64_t m, const int64_t packed_n,
                       const int64_t pack_num, const int64_t packed_stride, T* matmul_wx,
                       T* matmul_vx, T* y) {
  // dispatch index type
  if (pack_num < (1 << 30)) {
    LaunchFusedGluForwardGpu<T, int32_t, act_type, pack_size>(
        stream, m, packed_n, pack_num, packed_stride, matmul_wx, matmul_vx, y);
  } else {
    LaunchFusedGluForwardGpu<T, int64_t, act_type, pack_size>(
        stream, m, packed_n, pack_num, packed_stride, matmul_wx, matmul_vx, y);
  }
}

template<typename T, ep::primitive::UnaryOp act_type, int32_t alignment,
         typename std::enable_if<alignment / sizeof(T) == 0, int>::type = 0>
void DispatchPackSize(ep::Stream* stream, const int64_t m, const int64_t n, const int64_t stride,
                      T* matmul_wx, T* matmul_vx, T* y) {
  DispatchIndexType<T, act_type, 1>(stream, m, n, m * n, stride, matmul_wx, matmul_vx, y);
}

template<typename T, ep::primitive::UnaryOp act_type, int32_t alignment,
         typename std::enable_if<alignment / sizeof(T) != 0, int>::type = 0>
void DispatchPackSize(ep::Stream* stream, const int64_t m, const int64_t n, const int64_t stride,
                      T* matmul_wx, T* matmul_vx, T* y) {
  const int64_t pack_size = alignment / sizeof(T);
  const int64_t packed_n = n / pack_size;
  const int64_t pack_num = m * packed_n;
  const int64_t packed_stride = stride / pack_size;
  DispatchIndexType<T, act_type, alignment / sizeof(T)>(stream, m, packed_n, pack_num,
                                                        packed_stride, matmul_wx, matmul_vx, y);
}

template<typename T, ep::primitive::UnaryOp act_type>
void DispatchAlignment(ep::Stream* stream, const int64_t m, const int64_t n, const int64_t stride,
                       T* matmul_wx, T* matmul_vx, T* y) {
  const auto IsAligned = [&](const size_t alignment) {
    const uintptr_t matmul_wx_ptr = reinterpret_cast<uintptr_t>(matmul_wx);
    const uintptr_t matmul_vx_ptr = reinterpret_cast<uintptr_t>(matmul_vx);
    const uintptr_t y_ptr = reinterpret_cast<uintptr_t>(y);

    return (/* memory address alignment */
            matmul_wx_ptr % alignment == 0 && matmul_vx_ptr % alignment == 0
            && y_ptr % alignment == 0
            /* #element per row alignment */
            && n % (alignment / sizeof(T)) == 0);
  };

  if (IsAligned(16)) {
    DispatchPackSize<T, act_type, 16>(stream, m, n, stride, matmul_wx, matmul_vx, y);
  } else if (IsAligned(8)) {
    DispatchPackSize<T, act_type, 8>(stream, m, n, stride, matmul_wx, matmul_vx, y);
  } else if (IsAligned(4)) {
    DispatchPackSize<T, act_type, 4>(stream, m, n, stride, matmul_wx, matmul_vx, y);
  } else if (IsAligned(2)) {
    DispatchPackSize<T, act_type, 2>(stream, m, n, stride, matmul_wx, matmul_vx, y);
  } else {
    DispatchPackSize<T, act_type, 1>(stream, m, n, stride, matmul_wx, matmul_vx, y);
  }
}

template<typename T>
void DispatchActivationType(ep::Stream* stream, const int64_t m, const int64_t n,
                            const int64_t stride, T* matmul_wx, T* matmul_vx, T* y,
                            const std::string& activation) {
  if (activation == "none") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kIdentity>(stream, m, n, stride, matmul_wx,
                                                            matmul_vx, y);
  } else if (activation == "sigmoid") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kSigmoid>(stream, m, n, stride, matmul_wx,
                                                           matmul_vx, y);
  } else if (activation == "relu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kRelu>(stream, m, n, stride, matmul_wx, matmul_vx,
                                                        y);
  } else if (activation == "gelu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kGelu>(stream, m, n, stride, matmul_wx, matmul_vx,
                                                        y);
  } else if (activation == "fast_gelu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kFastGelu>(stream, m, n, stride, matmul_wx,
                                                            matmul_vx, y);
  } else if (activation == "silu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kSilu>(stream, m, n, stride, matmul_wx, matmul_vx,
                                                        y);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
class GpuFusedGluKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  GpuFusedGluKernel() = default;
  ~GpuFusedGluKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    // obtain tensors from context
    const user_op::Tensor* input_tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* input_tensor_w = ctx->Tensor4ArgNameAndIndex("w", 0);
    user_op::Tensor* out_tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out_tensor_matmul_wx = ctx->Tensor4ArgNameAndIndex("matmul_wx", 0);

    // obtain optional tensors from context
    bool is_split_mode = false;
    user_op::Tensor* input_tensor_b = nullptr;
    user_op::Tensor* input_tensor_v = nullptr;
    user_op::Tensor* input_tensor_c = nullptr;
    user_op::Tensor* out_tensor_matmul_vx = nullptr;

    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    const auto* fused_glu_cache =
        CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));

    // check whether the user provide weight tensor v
    if (ctx->has_input("v", 0)) {
      input_tensor_v = ctx->Tensor4ArgNameAndIndex("v", 0);
      out_tensor_matmul_vx = ctx->Tensor4ArgNameAndIndex("matmul_vx", 0);
      is_split_mode = true;
    }

    bool has_b = ctx->has_input("b", 0);
    bool has_c = ctx->has_input("c", 0);

    // check whether the user provide bais tensors
    CHECK(!(has_b && (is_split_mode && !has_c)))
        << "expected existance of c, when provide tensors w, v and b";
    bool has_bias = false;
    if (has_b && (is_split_mode && has_c)) {
      input_tensor_b = ctx->Tensor4ArgNameAndIndex("b", 0);
      input_tensor_c = ctx->Tensor4ArgNameAndIndex("c", 0);
      has_bias = true;
    } else if (has_b && (!is_split_mode)) {
      input_tensor_b = ctx->Tensor4ArgNameAndIndex("b", 0);
      has_bias = true;
    } else {
      has_bias = false;
    }

    cublasLtEpilogue_t epilogue;
    if (has_bias) {
      epilogue = CUBLASLT_EPILOGUE_BIAS;
    } else {
      epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }

    // obtain tensor shapes
    const ShapeView& x_shape = input_tensor_x->shape_view();
    const ShapeView& w_shape = input_tensor_w->shape_view();
    ShapeView b_shape;
    if (has_bias) {
      Shape _b_shape;
      input_tensor_b->shape_view().ToShape(&_b_shape);
      b_shape = ShapeView(_b_shape);
    }
    const ShapeView& y_shape = out_tensor_y->shape_view();

    // validate dimension and number of axes
    CHECK_GT(x_shape.NumAxes(), 1)
        << "number of axes of \'x\' should have be greater than 1, yet get " << x_shape.NumAxes();
    CHECK_EQ(w_shape.NumAxes(), 2)
        << "number of axes of \'w\' should have be equal to 2, yet get " << w_shape.NumAxes();
    if (has_bias) {
      CHECK_EQ(b_shape.NumAxes(), 1)
          << "number of axes of \'b\' should have be equal to 1, yet get " << b_shape.NumAxes();
    }

    // check input tensor shapes
    size_t x_num_axes = x_shape.NumAxes();
    CHECK_EQ(w_shape.At(1), x_shape.At(x_num_axes - 1))
        << "dimension 1 of \'w\'(" << w_shape.At(1)
        << ") is not consistant with the last dimension of \'x\'(" << x_shape.At(x_num_axes - 1)
        << ")";
    if (has_bias) {
      CHECK_EQ(b_shape.At(0), w_shape.At(0))
          << "dimension 0 of \'b\'(" << b_shape.At(0)
          << ") is not consistant with dimension 0 of \'w\'(" << w_shape.At(0) << ")";
    }
    if (!is_split_mode) {
      CHECK_EQ(w_shape.At(1) % 2, 0) << "dimension 1 of \'w\' is not divisible by 2";
    }

    // check optional input tensor shapes
    if (is_split_mode) {
      const ShapeView& v_shape = input_tensor_v->shape_view();
      CHECK_EQ(v_shape.NumAxes(), 2)
          << "number of axes of \'v\' should have be equal to 2, yet get " << v_shape.NumAxes();
      CHECK_EQ(v_shape, w_shape) << "the shape of \'v\' is not consistant with \'w\'";
      if (has_bias) {
        const ShapeView& c_shape = input_tensor_c->shape_view();
        CHECK_EQ(c_shape.NumAxes(), 1)
            << "number of axes of \'c\' should have be equal to 1, yet get " << c_shape.NumAxes();
        CHECK_EQ(c_shape, b_shape) << "the shape of \'c\' is not consistant with \'b\'";
      }
    }

    // obtain data type for cublaslt computation
    const DataType data_type = out_tensor_matmul_wx->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);

    // infer m, n, k
    const int64_t m = x_shape.Count(0, x_num_axes - 1);
    const int64_t n = y_shape.At(x_num_axes - 1);
    const int64_t k = x_shape.At(x_num_axes - 1);

    if (has_bias) {
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
    }

    // init scalar parameters for cublaslt
    const double alpha = 1.0;
    const double beta = 0.0;
    const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    // calculate matmul_wx (and matmul_vx) through cublaslt
    if (is_split_mode) {
      // define shape parameters to be inferred
      size_t cublas_wx_m = 0, cublas_wx_n = 0, cublas_wx_k = 0;
      int64_t cublas_wx_lda = 0, cublas_wx_ldb = 0, cublas_wx_ldc = 0;
      size_t cublas_vx_m = 0, cublas_vx_n = 0, cublas_vx_k = 0;
      int64_t cublas_vx_lda = 0, cublas_vx_ldb = 0, cublas_vx_ldc = 0;

      // init dim vector
      DimVector x_dim_vec({m, k});
      DimVector w_dim_vec({n, k});
      DimVector v_dim_vec({n, k});

      // setup cublaslt matmul attributes
      InferMatmulCublasMNK(x_dim_vec, w_dim_vec,
                           /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                           /*transpose_b=*/ep::primitive::BlasTransposeType::T, &cublas_wx_m,
                           &cublas_wx_n, &cublas_wx_k, &cublas_wx_lda, &cublas_wx_ldb,
                           &cublas_wx_ldc);
      SetCublasAttr(fused_glu_cache, cublas_compute_dtype, cuda_data_type, false,
                    /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                    /*transpose_b=*/ep::primitive::BlasTransposeType::T, epilogue,
                    has_bias ? input_tensor_b->dptr() : nullptr, nullptr, cublas_wx_m, cublas_wx_n,
                    cublas_wx_k, cublas_wx_lda, cublas_wx_ldb, cublas_wx_ldc);

      // setup algorithms
      cublasLtMatmulPreference_t preference = nullptr;
      size_t workspace_size = cuda_stream->cublas_workspace_size();
      OF_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
      OF_CUBLAS_CHECK(
          cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_size, sizeof(workspace_size)));
      int wx_returned_result = 0;
      cublasLtMatmulHeuristicResult_t wx_heuristic_result;
      OF_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
          cuda_stream->cublas_lt_handle(), fused_glu_cache->operation_desc,
          fused_glu_cache->cublas_a_desc, fused_glu_cache->cublas_b_desc,
          fused_glu_cache->cublas_c_desc, fused_glu_cache->cublas_c_desc, preference, 1,
          &wx_heuristic_result, &wx_returned_result));
      CHECK_EQ(wx_returned_result, 1);

      // launch cublaslt matmul
      // out_tensor_matmul_wx = 1.0 * (input_tensor_w * input_tensor_x) + 1.0 * input_tensor_b
      OF_CUBLAS_CHECK(cublasLtMatmul(
          /*lightHandle*/ cuda_stream->cublas_lt_handle(),
          /*computeDesc*/ fused_glu_cache->operation_desc,
          /*alpha*/ &sp_alpha,
          /*A*/ input_tensor_w->dptr(),
          /*Adesc*/ fused_glu_cache->cublas_a_desc,
          /*B*/ input_tensor_x->dptr(),
          /*Bdesc*/ fused_glu_cache->cublas_b_desc,
          /*beta*/ &sp_beta,
          /*C*/ has_bias ? input_tensor_b->dptr() : nullptr,
          /*Cdesc*/ fused_glu_cache->cublas_c_desc,
          /*D*/ out_tensor_matmul_wx->mut_dptr(),
          /*Ddesc*/ fused_glu_cache->cublas_c_desc,
          /*algo*/ &wx_heuristic_result.algo,
          /*workspace*/ cuda_stream->cublas_workspace(),
          /*workspaceSizeInBytes*/ cuda_stream->cublas_workspace_size(),
          /*stream*/ cuda_stream->cuda_stream()));

      // setup cublaslt attributes
      InferMatmulCublasMNK(x_dim_vec, v_dim_vec,
                           /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                           /*transpose_b=*/ep::primitive::BlasTransposeType::T, &cublas_vx_m,
                           &cublas_vx_n, &cublas_vx_k, &cublas_vx_lda, &cublas_vx_ldb,
                           &cublas_vx_ldc);
      SetCublasAttr(fused_glu_cache, cublas_compute_dtype, cuda_data_type, false,
                    /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                    /*transpose_b=*/ep::primitive::BlasTransposeType::T, epilogue,
                    has_bias ? input_tensor_c->dptr() : nullptr, nullptr, cublas_vx_m, cublas_vx_n,
                    cublas_vx_k, cublas_vx_lda, cublas_vx_ldb, cublas_vx_ldc);

      // setup algorithm
      int vx_returned_result = 0;
      cublasLtMatmulHeuristicResult_t vx_heuristic_result;
      OF_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
          cuda_stream->cublas_lt_handle(), fused_glu_cache->operation_desc,
          fused_glu_cache->cublas_a_desc, fused_glu_cache->cublas_b_desc,
          fused_glu_cache->cublas_c_desc, fused_glu_cache->cublas_c_desc, preference, 1,
          &vx_heuristic_result, &vx_returned_result));
      CHECK_EQ(vx_returned_result, 1);
      cublasLtMatmulPreferenceDestroy(preference);

      // launch cublaslt matmul
      // out_tensor_matmul_vx = 1.0 * (input_tensor_v * input_tensor_x) + 1.0 * input_tensor_c
      OF_CUBLAS_CHECK(cublasLtMatmul(
          /*lightHandle*/ cuda_stream->cublas_lt_handle(),
          /*computeDesc*/ fused_glu_cache->operation_desc,
          /*alpha*/ &sp_alpha,
          /*A*/ input_tensor_v->dptr(),
          /*Adesc*/ fused_glu_cache->cublas_a_desc,
          /*B*/ input_tensor_x->dptr(),
          /*Bdesc*/ fused_glu_cache->cublas_b_desc,
          /*beta*/ &sp_beta,
          /*C*/ has_bias ? input_tensor_c->dptr() : nullptr,
          /*Cdesc*/ fused_glu_cache->cublas_c_desc,
          /*D*/ out_tensor_matmul_vx->mut_dptr(),
          /*Ddesc*/ fused_glu_cache->cublas_c_desc,
          /*algo*/ &wx_heuristic_result.algo,
          /*workspace*/ cuda_stream->cublas_workspace(),
          /*workspaceSizeInBytes*/ cuda_stream->cublas_workspace_size(),
          /*stream*/ cuda_stream->cuda_stream()));
    } else {
      // define shape parameters to be inferred
      size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
      int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

      // init dim vector
      DimVector x_dim_vec({m, k});
      DimVector w_dim_vec({2 * n, k});

      // setup cublas attributes
      InferMatmulCublasMNK(x_dim_vec, w_dim_vec,
                           /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                           /*transpose_b=*/ep::primitive::BlasTransposeType::T, &cublas_m,
                           &cublas_n, &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);
      SetCublasAttr(fused_glu_cache, cublas_compute_dtype, cuda_data_type, false,
                    /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                    /*transpose_b=*/ep::primitive::BlasTransposeType::T, epilogue,
                    has_bias ? input_tensor_b->dptr() : nullptr, nullptr, cublas_m, cublas_n,
                    cublas_k, cublas_lda, cublas_ldb, cublas_ldc);

      // setup algorithm
      cublasLtMatmulPreference_t preference = nullptr;
      size_t workspace_size = cuda_stream->cublas_workspace_size();
      OF_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
      OF_CUBLAS_CHECK(
          cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_size, sizeof(workspace_size)));
      int wx_returned_result = 0;
      cublasLtMatmulHeuristicResult_t wx_heuristic_result;
      OF_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
          cuda_stream->cublas_lt_handle(), fused_glu_cache->operation_desc,
          fused_glu_cache->cublas_a_desc, fused_glu_cache->cublas_b_desc,
          fused_glu_cache->cublas_c_desc, fused_glu_cache->cublas_c_desc, preference, 1,
          &wx_heuristic_result, &wx_returned_result));
      CHECK_EQ(wx_returned_result, 1);
      cublasLtMatmulPreferenceDestroy(preference);

      // launch cublaslt matmul
      // out_tensor_matmul_wx = 1.0 * (input_tensor_w * input_tensor_x) + 1.0 * input_tensor_b
      OF_CUBLAS_CHECK(cublasLtMatmul(
          /*lightHandle*/ cuda_stream->cublas_lt_handle(),
          /*computeDesc*/ fused_glu_cache->operation_desc,
          /*alpha*/ &sp_alpha,
          /*A*/ input_tensor_w->dptr(),
          /*Adesc*/ fused_glu_cache->cublas_a_desc,
          /*B*/ input_tensor_x->dptr(),
          /*Bdesc*/ fused_glu_cache->cublas_b_desc,
          /*beta*/ &sp_beta,
          /*C*/ has_bias ? input_tensor_b->dptr() : nullptr,
          /*Cdesc*/ fused_glu_cache->cublas_c_desc,
          /*D*/ out_tensor_matmul_wx->mut_dptr(),
          /*Ddesc*/ fused_glu_cache->cublas_c_desc,
          /*algo*/ nullptr,
          /*workspace*/ cuda_stream->cublas_workspace(),
          /*workspaceSizeInBytes*/ cuda_stream->cublas_workspace_size(),
          /*stream*/ cuda_stream->cuda_stream()));
    }

    // dispatch according to activation type
    DispatchActivationType<T>(ctx->stream(),
                              /*m, n=*/m, n,
                              /*stride=*/is_split_mode ? n : 2 * n,
                              /*matmul_wx=*/out_tensor_matmul_wx->mut_dptr<T>(),
                              /*matmul_vx=*/
                              is_split_mode ? out_tensor_matmul_vx->mut_dptr<T>()
                                            : out_tensor_matmul_wx->mut_dptr<T>() + n,
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

#endif  // CUDA_VERSION >= 11020
