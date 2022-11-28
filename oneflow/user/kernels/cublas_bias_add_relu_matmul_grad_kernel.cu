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
class CublasBiasAddReluMatmulGradKernel final : public user_op::OpKernel,
                                                public user_op::CudaGraphSupport {
 public:
  CublasBiasAddReluMatmulGradKernel() = default;
  ~CublasBiasAddReluMatmulGradKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* aux = ctx->Tensor4ArgNameAndIndex("aux", 0);
    user_op::Tensor* d_bias = ctx->Tensor4ArgNameAndIndex("d_bias", 0);
    user_op::Tensor* d_grad = ctx->Tensor4ArgNameAndIndex("d_grad", 0);
    const auto* matmul_grad_cache =
        CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    const DataType data_type = dy->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

    const double alpha = ctx->Attr<double>("alpha");
    const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    const double beta = 0.0;
    const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    // currently only support 2D matmul.
    DimVector dy_shape(2);
    dy->shape_view().ToDimVector(&dy_shape);
    DimVector weight_shape(2);
    weight->shape_view().ToDimVector(&weight_shape);
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DRELU_BGRAD;

    InferMatmulCublasMNK(dy_shape, weight_shape,
                         /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                         /*transpose_b=*/ep::primitive::BlasTransposeType::N, &cublas_m, &cublas_n,
                         &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);

    SetCublasAttr(matmul_grad_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/true,
                  /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                  /*transpose_b=*/ep::primitive::BlasTransposeType::N, epilogue, d_bias->dptr(),
                  aux->dptr(), cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc);
    /*
    a = dy, b = weight
    cublas_a=weight, cublas_b=dy
    */
    OF_CUBLAS_CHECK(
        cublasLtMatmul(cuda_stream->cublas_lt_handle(), matmul_grad_cache->operation_desc,
                       &sp_alpha, weight->dptr(), matmul_grad_cache->cublas_a_desc, dy->dptr(),
                       matmul_grad_cache->cublas_b_desc, &sp_beta, d_grad->mut_dptr(),
                       matmul_grad_cache->cublas_c_desc, d_grad->mut_dptr(),
                       matmul_grad_cache->cublas_c_desc, nullptr, cuda_stream->cublas_workspace(),
                       cuda_stream->cublas_workspace_size(), cuda_stream->cuda_stream()));
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUBLAS_BIAS_ADD_RELU_MATMUL_GRAD_KERNEL(dtype)        \
  REGISTER_USER_KERNEL("cublas_bias_add_relu_matmul_grad")             \
      .SetCreateFn<CublasBiasAddReluMatmulGradKernel<dtype>>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("weight", 0) == GetDataType<dtype>::value));

REGISTER_CUBLAS_BIAS_ADD_RELU_MATMUL_GRAD_KERNEL(float)
REGISTER_CUBLAS_BIAS_ADD_RELU_MATMUL_GRAD_KERNEL(double)
REGISTER_CUBLAS_BIAS_ADD_RELU_MATMUL_GRAD_KERNEL(half)

}  // namespace

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11060
