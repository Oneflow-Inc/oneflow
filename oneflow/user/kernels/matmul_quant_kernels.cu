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

namespace oneflow {

namespace {

using RowMajor = cutlass::layout::RowMajor;     // 行主序存储方式
using ColMajor = cutlass::layout::ColumnMajor;  // 列主序存储方式

void cutlass_gemm_scale_bias_s8s8fp16(cudaStream_t stream, void* workspace, int m, int k, int n,
                                      const int8_t* a_ptr, const int8_t* b_ptr,
                                      const cutlass::half_t* scale, const cutlass::half_t* bias,
                                      cutlass::half_t* output) {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = cutlass::half_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using CutlassRowAColBRowCGemm = typename cutlass::gemm::device::GemmScaleBiasFusion<
      ElementA,  // A矩阵数据类型
      RowMajor,  // A矩阵存储方式
      ElementB,  // B矩阵数据类型
      ColMajor,  // B矩阵存储方式
      ElementC,  // C矩阵数据类型
      RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<8, 8, 16>,
      cutlass::epilogue::thread::LinearCombinationScaleBias<ElementC, 8, ElementAccumulator,
                                                            ElementCompute>>;

  CutlassRowAColBRowCGemm gemm_operator;
  CutlassRowAColBRowCGemm::Arguments args({m, n, k}, {a_ptr, k}, {b_ptr, k}, {scale, 0}, {bias, 0},
                                          {output, n});

  cutlass::Status init_status = gemm_operator.initialize(args, workspace, stream);
  CHECK(init_status == cutlass::Status::kSuccess);
  auto run_status = gemm_operator(stream);  //运行Gemm
  CHECK(run_status == cutlass::Status::kSuccess);
  return;
}

void cutlass_gemm_scale_bias_s8s8fp32(cudaStream_t stream, void* workspace, int m, int k, int n,
                                      const int8_t* a_ptr, const int8_t* b_ptr, const float* scale,
                                      const float* bias, float* output) {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = float;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using CutlassRowAColBRowCGemm = typename cutlass::gemm::device::GemmScaleBiasFusion<
      ElementA,  // A矩阵数据类型
      RowMajor,  // A矩阵存储方式
      ElementB,  // B矩阵数据类型
      ColMajor,  // B矩阵存储方式
      ElementC,  // C矩阵数据类型
      RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<8, 8, 16>,
      cutlass::epilogue::thread::LinearCombinationScaleBias<ElementC, 8, ElementAccumulator,
                                                            ElementCompute>>;

  CutlassRowAColBRowCGemm gemm_operator;
  CutlassRowAColBRowCGemm::Arguments args({m, n, k}, {a_ptr, k}, {b_ptr, k}, {scale, 0}, {bias, 0},
                                          {output, n});

  cutlass::Status init_status = gemm_operator.initialize(args, workspace, stream);
  CHECK(init_status == cutlass::Status::kSuccess);
  auto run_status = gemm_operator(stream);  //运行Gemm
  CHECK(run_status == cutlass::Status::kSuccess);
  return;
}

}  // namespace

class MatmulQuantKernel final : public user_op::OpKernel {
 public:
  MatmulQuantKernel() = default;
  ~MatmulQuantKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    const user_op::Tensor* scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
    CHECK(add_to_output == nullptr);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    CHECK(ctx->Attr<bool>("transpose_b"));

    int64_t dim_a = a->shape_view().NumAxes();
    const int m = a->shape_view().Count(0, dim_a - 1);
    const int k = a->shape_view().At(dim_a - 1);
    const int n = b->shape_view().At(0);

    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    if (out->data_type() == DataType::kFloat) {
      cutlass_gemm_scale_bias_s8s8fp32(ctx->stream()->As<ep::CudaStream>()->cuda_stream(),
                                       tmp_buffer->mut_dptr(), m, k, n, a->dptr<int8_t>(),
                                       b->dptr<int8_t>(), scale->dptr<float>(), bias->dptr<float>(),
                                       out->mut_dptr<float>());
    } else if (out->data_type() == DataType::kFloat16) {
      cutlass_gemm_scale_bias_s8s8fp16(ctx->stream()->As<ep::CudaStream>()->cuda_stream(),
                                       tmp_buffer->mut_dptr(), m, k, n, a->dptr<int8_t>(),
                                       b->dptr<int8_t>(),
                                       reinterpret_cast<const cutlass::half_t*>(scale->dptr()),
                                       reinterpret_cast<const cutlass::half_t*>(bias->dptr()),
                                       reinterpret_cast<cutlass::half_t*>(out->mut_dptr()));
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
