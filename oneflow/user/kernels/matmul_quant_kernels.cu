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

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

#include "cutlass/gemm/device/gemm.h"

#include "cutlass/gemm/device/gemm_scale_bias_fusion.h"
#include "cutlass/epilogue/thread/linear_combination_scale_bias.h"

#include "cutlass/gemm/device/gemm_scale_bias_residual_fusion.h"
#include "cutlass/epilogue/thread/linear_combination_scale_bias_residual.h"

namespace oneflow {

namespace {

using RowMajor = cutlass::layout::RowMajor;
using ColMajor = cutlass::layout::ColumnMajor;

template<typename T, typename OutT>
void cutlass_gemm_scale_bias_s8(cudaStream_t stream, void* workspace, int m, int n, int k,
                                const void* a, const void* b, const void* scale, const void* bias,
                                const void* residual, void* output) {
  using ElementA = T;
  using ElementB = T;
  using ElementC = OutT;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  if (!residual) {
    using CutlassRowAColBRowCGemm = typename cutlass::gemm::device::GemmScaleBiasFusion<
        ElementA, RowMajor, ElementB, ColMajor, ElementC, RowMajor, ElementAccumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75, cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<8, 8, 16>,
        cutlass::epilogue::thread::LinearCombinationScaleBias<ElementC, 8, ElementAccumulator,
                                                              ElementCompute>>;

    CutlassRowAColBRowCGemm gemm_operator;
    typename CutlassRowAColBRowCGemm::Arguments args(
        {m, n, k}, {reinterpret_cast<const T*>(a), k}, {reinterpret_cast<const T*>(b), k},
        {reinterpret_cast<const OutT*>(scale), 0}, {reinterpret_cast<const OutT*>(bias), 0},
        {reinterpret_cast<OutT*>(output), n});

    cutlass::Status init_status = gemm_operator.initialize(args, workspace, stream);
    CHECK(init_status == cutlass::Status::kSuccess);
    auto run_status = gemm_operator(stream);
    CHECK(run_status == cutlass::Status::kSuccess);
  } else {
    using CutlassRowAColBRowCGemm = typename cutlass::gemm::device::GemmScaleBiasResidualFusion<
        ElementA, RowMajor, ElementB, ColMajor, ElementC, RowMajor, ElementAccumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75, cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<8, 8, 16>,
        cutlass::epilogue::thread::LinearCombinationScaleBiasResidual<
            ElementC, 8, ElementAccumulator, ElementCompute, cutlass::plus>>;

    CutlassRowAColBRowCGemm gemm_operator;
    typename CutlassRowAColBRowCGemm::Arguments args(
        {m, n, k}, {reinterpret_cast<const T*>(a), k}, {reinterpret_cast<const T*>(b), k},
        {reinterpret_cast<const OutT*>(scale), 0}, {reinterpret_cast<const OutT*>(bias), 0},
        {reinterpret_cast<const OutT*>(residual), n}, {reinterpret_cast<OutT*>(output), n});

    cutlass::Status init_status = gemm_operator.initialize(args, workspace, stream);
    CHECK(init_status == cutlass::Status::kSuccess);
    auto run_status = gemm_operator(stream);
    CHECK(run_status == cutlass::Status::kSuccess);
  }
}

}  // namespace

class MatmulQuantKernel final : public user_op::OpKernel {
 public:
  MatmulQuantKernel() = default;
  ~MatmulQuantKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    const user_op::Tensor* scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    CHECK(!ctx->Attr<bool>("transpose_a"));
    CHECK(ctx->Attr<bool>("transpose_b"));

    int64_t dim_a = a->shape_view().NumAxes();
    const int m = a->shape_view().Count(0, dim_a - 1);
    const int k = a->shape_view().At(dim_a - 1);
    const int n = b->shape_view().At(0);

    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    if (out->data_type() == DataType::kFloat) {
      cutlass_gemm_scale_bias_s8<int8_t, float>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), tmp_buffer->mut_dptr(), m, n, k,
          a->dptr(), b->dptr(), scale->dptr(), bias->dptr(),
          (add_to_output ? add_to_output->dptr() : nullptr), out->mut_dptr());
    } else if (out->data_type() == DataType::kFloat16) {
      cutlass_gemm_scale_bias_s8<int8_t, cutlass::half_t>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), tmp_buffer->mut_dptr(), m, n, k,
          a->dptr(), b->dptr(), scale->dptr(), bias->dptr(),
          (add_to_output ? add_to_output->dptr() : nullptr), out->mut_dptr());
    }
  }
};

REGISTER_USER_KERNEL("matmul_quant")
    .SetCreateFn<MatmulQuantKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("a", 0) == DataType::kInt8)
                     && (user_op::HobDataType("b", 0) == DataType::kInt8))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {
      // use static workspace size
      return 128 * 1024 * 1024;
    })
    .SetPriority(user_op::kKernelPriorityOptimized);

}  // namespace oneflow

#endif  // WITH_CUTLASS_EXTENSION
