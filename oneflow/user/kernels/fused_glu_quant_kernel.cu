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
#ifdef WITH_CUTLASS_EXTENSION

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

#include "oneflow/user/kernels/cutlass_gemm_tuner.h"

#include <cutlass/library/library.h>
#include <cutlass/library/operation_table.h>
#include <cutlass/library/cutlass_extension_library.h>
#include <nlohmann/json.hpp>

#if CUDA_VERSION >= 11020

namespace oneflow {

namespace {

void LaunchMatmulQuantOp(user_op::KernelComputeContext* ctx,
                         const cutlass::library::GemmFunctionalKey& key,
                         const cutlass::gemm::GemmCoord& problem_size, const user_op::Tensor* a,
                         const user_op::Tensor* b, const user_op::Tensor* in_zero_point,
                         const user_op::Tensor* in_scale, const user_op::Tensor* weight_scale,
                         const user_op::Tensor* weight_acc, const user_op::Tensor* scale,
                         const user_op::Tensor* bias, const user_op::Tensor* add_to_output,
                         user_op::Tensor* out) {
  cutlass::library::GemmScaleBiasFusionConfiguration configuraion;
  configuraion.problem_size = problem_size;
  configuraion.lda = problem_size.k();
  configuraion.ldb = problem_size.k();
  configuraion.ld_filter_scale = 0;
  configuraion.ld_filter_acc = 0;
  configuraion.ld_scale = 0;
  configuraion.ld_bias = 0;
  configuraion.ldr = problem_size.n();
  configuraion.ldd = problem_size.n();
  configuraion.split_k_slices = 1;
  // if (problem_size.m() <= 2 && problem_size.k() >= 4096) { configuraion.split_k_slices = 16; }

  cutlass::library::GemmScaleBiasFusionArguments arguments;
  arguments.A = a->dptr();
  arguments.B = b->dptr();
  arguments.D = out->mut_dptr();
  arguments.P = nullptr;
  arguments.InScale = nullptr;
  arguments.FilterScale = nullptr;
  arguments.FilterAcc = nullptr;
  arguments.Scale = nullptr;
  arguments.Bias = nullptr;
  arguments.Residual = nullptr;

  if (in_zero_point) { arguments.P = in_zero_point->dptr(); }
  if (in_scale) { arguments.InScale = in_scale->dptr(); }
  if (weight_scale) { arguments.FilterScale = weight_scale->dptr(); }
  if (weight_acc) { arguments.FilterAcc = weight_acc->dptr(); }
  if (scale) { arguments.Scale = scale->dptr(); }
  if (bias) { arguments.Bias = bias->dptr(); }
  if (add_to_output) { arguments.Residual = add_to_output->dptr(); }

  user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
  auto* stream = ctx->stream()->As<ep::CudaStream>();
  const cutlass::library::Operation* operation = nullptr;
  operation = [&]() -> const cutlass::library::Operation* {
    const std::string& tuning_cache = ctx->Attr<std::string>("tuning_cache");
    if (tuning_cache.empty()) { return nullptr; }
    auto tuning_cache_object = nlohmann::json::parse(tuning_cache);
    if (!tuning_cache_object.is_object()) { return nullptr; }
    auto it = tuning_cache_object.find("cutlass");
    if (it == tuning_cache_object.end()) { return nullptr; }
    if (!it->is_string()) { return nullptr; }
    const std::string name = *it;
    return CutlassGemmTuner().GetOperation(name, stream, key, configuraion, arguments,
                                           tmp_buffer->mut_dptr(),
                                           tmp_buffer->shape_view().elem_cnt());
  }();
  if (!operation) {
    operation = CutlassGemmTuner().FindOperation(stream, key, configuraion, arguments,
                                                 tmp_buffer->mut_dptr(),
                                                 tmp_buffer->shape_view().elem_cnt());
  }
  CHECK(operation != nullptr);
  const size_t host_workspace_size = operation->get_host_workspace_size(&configuraion);
  std::vector<uint8_t> host_workspace(host_workspace_size, 0);
  auto init_status = operation->initialize(&configuraion, host_workspace.data(),
                                           tmp_buffer->mut_dptr(), stream->cuda_stream());
  CHECK(init_status == cutlass::Status::kSuccess);
  auto run_status = operation->run(&arguments, host_workspace.data(), tmp_buffer->mut_dptr(),
                                   stream->cuda_stream());
  CHECK(run_status == cutlass::Status::kSuccess);
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

template<typename T, typename OutT>
class GpuFusedGluQuantKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  GpuFusedGluQuantKernel() = default;
  ~GpuFusedGluQuantKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* input_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* input_w = ctx->Tensor4ArgNameAndIndex("w", 0);
    user_op::Tensor* out_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out_matmul_wx = ctx->Tensor4ArgNameAndIndex("matmul_wx", 0);
    user_op::Tensor* out_matmul_vx = nullptr;

    CHECK(!ctx->has_input("v", 0)) << "fused_glu_quant does not support split mode";
    bool is_split_mode = false;

    const ShapeView& x_shape = input_x->shape_view();
    const ShapeView& w_shape = input_w->shape_view();
    const ShapeView& y_shape = out_y->shape_view();

    const DataType data_type = out_y->data_type();
    size_t x_num_axes = x_shape.NumAxes();
    // infer m, n, k
    const int64_t m = x_shape.Count(0, x_num_axes - 1);
    const int64_t n = y_shape.At(x_num_axes - 1);
    const int64_t k = x_shape.At(x_num_axes - 1);

    cutlass::library::GemmFunctionalKey key(
        cutlass::library::Provider::kCUTLASS, cutlass::library::GemmKind::kGemm,
        cutlass::library::NumericTypeID::kS32,         // element_compute
        cutlass::library::NumericTypeID::kS32,         // element_scalar
        cutlass::library::NumericTypeID::kS8,          // element_A
        cutlass::library::LayoutTypeID::kRowMajor,     // layout_A
        cutlass::library::ComplexTransform::kNone,     // transform_A
        cutlass::library::NumericTypeID::kS8,          // element_B
        cutlass::library::LayoutTypeID::kColumnMajor,  // layout_B
        cutlass::library::ComplexTransform::kNone,     // transform_B
        cutlass::library::NumericTypeID::kS32,         // element_C
        cutlass::library::LayoutTypeID::kRowMajor,     // layout_C
        cutlass::library::NumericTypeID::kS32,         // element_D
        cutlass::library::LayoutTypeID::kRowMajor      // layout_D
    );
    if (data_type == DataType::kFloat) {
      key.element_scalar = cutlass::library::NumericTypeID::kF32;
      key.element_C = cutlass::library::NumericTypeID::kF32;
      key.element_D = cutlass::library::NumericTypeID::kF32;
    } else if (data_type == DataType::kFloat16) {
      key.element_scalar = cutlass::library::NumericTypeID::kF32;
      key.element_C = cutlass::library::NumericTypeID::kF16;
      key.element_D = cutlass::library::NumericTypeID::kF16;
    }
    cutlass::gemm::GemmCoord problem_size(m, 2 * n, k);

    const user_op::Tensor* in_zero_point = ctx->Tensor4ArgNameAndIndex("in_zero_point", 0);
    const user_op::Tensor* in_scale = ctx->Tensor4ArgNameAndIndex("in_scale", 0);
    const user_op::Tensor* weight_scale = ctx->Tensor4ArgNameAndIndex("weight_scale", 0);
    const user_op::Tensor* weight_acc = ctx->Tensor4ArgNameAndIndex("weight_acc", 0);
    const user_op::Tensor* input_scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    const user_op::Tensor* input_bias = ctx->Tensor4ArgNameAndIndex("bias", 0);

    LaunchMatmulQuantOp(ctx, key, problem_size, input_x, input_w, in_zero_point, in_scale,
                        weight_scale, weight_acc, input_scale, input_bias, nullptr, out_matmul_wx);

    // dispatch according to activation type
    DispatchActivationType<OutT>(
        ctx->stream(),
        /*m, n=*/m, n,
        /*stride=*/is_split_mode ? n : 2 * n,
        /*matmul_wx=*/out_matmul_wx->mut_dptr<OutT>(),
        /*matmul_vx=*/
        is_split_mode ? out_matmul_vx->mut_dptr<OutT>() : out_matmul_wx->mut_dptr<OutT>() + n,
        /*y=*/out_y->mut_dptr<OutT>(),
        /*activation=*/ctx->Attr<std::string>("activation"));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_GPU_FUSED_GLU_QUANT_KERNEL(T, OutT)                                  \
  REGISTER_USER_KERNEL("fused_glu_quant")                                             \
      .SetCreateFn<GpuFusedGluQuantKernel<T, OutT>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                \
                       && (user_op::HobDataType("x", 0) == GetDataType<T>::value)     \
                       && (user_op::HobDataType("y", 0) == GetDataType<OutT>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t { return 128 * 1024 * 1024; })

REGISTER_GPU_FUSED_GLU_QUANT_KERNEL(int8_t, float);
REGISTER_GPU_FUSED_GLU_QUANT_KERNEL(int8_t, half);

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11020

#endif  // WITH_CUTLASS_EXTENSION
