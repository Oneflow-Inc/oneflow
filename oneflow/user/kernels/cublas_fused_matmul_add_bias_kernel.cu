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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/cublas_fused_mlp_util.cuh"
// CUBLAS_AUX_EPILOGUE only support in cuda11.4 or higher version, in cuda11.4 it need static link.
#if CUDA_VERSION >= 11060

namespace oneflow {

namespace {

template<typename T>
class CublasFusedMatmulAddBiasKernel final : public user_op::OpKernel,
                                             public user_op::CudaGraphSupport {
 public:
  CublasFusedMatmulAddBiasKernel() = default;
  ~CublasFusedMatmulAddBiasKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    const auto* matmul_cache = CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const DataType data_type = out->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

    const double alpha = 1.0;
    const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    const double beta = 0.0;
    const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    // Currently only support 2D matmul.
    DimVector in_shape(x->shape_view().NumAxes());
    x->shape_view().ToDimVector(&in_shape);

    DimVector weight_shape(2);

    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);

    weight->shape_view().ToDimVector(&weight_shape);

    InferMatmulCublasMNK(in_shape, weight_shape,
                         /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                         /*transpose_b=*/ep::primitive::BlasTransposeType::T, &cublas_m, &cublas_n,
                         &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    T* y_ptr = nullptr;

    SetCublasAttr(matmul_cache, cublas_compute_dtype, cuda_data_type, false,
                  /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                  /*transpose_b=*/ep::primitive::BlasTransposeType::T, epilogue, bias->dptr(),
                  nullptr, cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc);

    int num_batches = 1;
    for (int i = 0; i < x->shape_view().NumAxes()-2; i++)
      num_batches *= x->shape_view().At(i);

    for (int i = 0; i < num_batches; i++) {
      y_ptr = static_cast<T*>(ctx->Tensor4ArgNameAndIndex("out", 0)->mut_dptr()) + cublas_m * cublas_n * i;
      OF_CUBLAS_CHECK(
          cublasLtMatmul(cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &sp_alpha,
                        weight->dptr(), matmul_cache->cublas_a_desc, static_cast<const T*>(x->dptr()) + cublas_n * cublas_k * i,
                        matmul_cache->cublas_b_desc, &sp_beta, y_ptr, matmul_cache->cublas_c_desc,
                        y_ptr, matmul_cache->cublas_c_desc, nullptr, cuda_stream->cublas_workspace(),
                        cuda_stream->cublas_workspace_size(), cuda_stream->cuda_stream()));
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUBLAS_FUSED_MATMUL_BIAS_KERNEL_GPU(cpp_type, data_type) \
  REGISTER_USER_KERNEL("cublas_fused_matmul_add_bias")                        \
      .SetCreateFn<CublasFusedMatmulAddBiasKernel<cpp_type>>()            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)    \
                       && (user_op::HobDataType("out", 0) == data_type));

REGISTER_CUBLAS_FUSED_MATMUL_BIAS_KERNEL_GPU(double, DataType::kDouble);
REGISTER_CUBLAS_FUSED_MATMUL_BIAS_KERNEL_GPU(float, DataType::kFloat);
REGISTER_CUBLAS_FUSED_MATMUL_BIAS_KERNEL_GPU(half, DataType::kFloat16);
REGISTER_CUBLAS_FUSED_MATMUL_BIAS_KERNEL_GPU(nv_bfloat16, DataType::kBFloat16);

}  // namespace

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11060
