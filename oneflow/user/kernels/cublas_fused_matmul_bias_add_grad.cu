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
#if CUDA_VERSION >= 11040
#include "stdio.h"

namespace oneflow {

template<typename T>
__global__ void printkernel(const T* in, int64_t n){
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x; 
  printf("Here is: %f \n", in[gid]); 
}



template<typename T>
class CublasMatmulBiasAddGradKernel final : public user_op::OpKernel,
                                                public user_op::CudaGraphSupport {
 public:
  CublasMatmulBiasAddGradKernel() = default;
  ~CublasMatmulBiasAddGradKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* w_grad = ctx->Tensor4ArgNameAndIndex("w_grad", 0);
    user_op::Tensor* b_grad = ctx->Tensor4ArgNameAndIndex("b_grad", 0);

    const auto* matmul_grad_cache =
        CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    const DataType data_type = dy->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

    const double alpha = 1.0;
    const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    const double beta = 0.0;
    const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    // currently only support 2D matmul.
    DimVector dy_shape(2);
    dy->shape().ToDimVector(&dy_shape);
    DimVector x_shape(2);
    x->shape().ToDimVector(&x_shape);
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BGRADB;

    InferMatmulCublasMNK(dy_shape, x_shape,
                         /*transpose_a=*/ep::primitive::BlasTransposeType::T,
                         /*transpose_b=*/ep::primitive::BlasTransposeType::N, &cublas_m, &cublas_n,
                         &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);
    
    printf("need to set bgrad ptr \n"); 
    SetCublasAttr(matmul_grad_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/false,
                  /*transpose_a=*/ep::primitive::BlasTransposeType::T,
                  /*transpose_b=*/ep::primitive::BlasTransposeType::N, epilogue, b_grad->mut_dptr(),
                  /*aux_ptr=*/nullptr, cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc);

    // printkernel<T><<<1, dy->shape().elem_cnt(), 0, cuda_stream->cuda_stream()>>>(dy->dptr<T>(), dy->shape().elem_cnt()); 
    printkernel<T><<<1, b_grad->shape().elem_cnt(), 0, cuda_stream->cuda_stream()>>>(b_grad->dptr<T>(), b_grad->shape().elem_cnt()); 
    /*
    a = dy, b = x
    cublas_a=x, cublas_b=dy
    */
    OF_CUBLAS_CHECK(
        cublasLtMatmul(cuda_stream->cublas_lt_handle(), matmul_grad_cache->operation_desc,
                       &sp_alpha, x->dptr(), matmul_grad_cache->cublas_a_desc, dy->dptr(),
                       matmul_grad_cache->cublas_b_desc, &sp_beta, w_grad->mut_dptr(),
                       matmul_grad_cache->cublas_c_desc, w_grad->mut_dptr(),
                       matmul_grad_cache->cublas_c_desc, nullptr, cuda_stream->cublas_workspace(),
                       cuda_stream->cublas_workspace_size(), cuda_stream->cuda_stream()));
    printkernel<T><<<1, b_grad->shape().elem_cnt(), 0, cuda_stream->cuda_stream()>>>(b_grad->dptr<T>(), b_grad->shape().elem_cnt()); 
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUBLAS_MATMUL_BIAS_ADD_GRAD_KERNEL(dtype)        \
  REGISTER_USER_KERNEL("cublas_matmul_bias_add_grad")             \
      .SetCreateFn<CublasMatmulBiasAddGradKernel<dtype>>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_CUBLAS_MATMUL_BIAS_ADD_GRAD_KERNEL(float)
REGISTER_CUBLAS_MATMUL_BIAS_ADD_GRAD_KERNEL(double)
REGISTER_CUBLAS_MATMUL_BIAS_ADD_GRAD_KERNEL(half)

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11040
