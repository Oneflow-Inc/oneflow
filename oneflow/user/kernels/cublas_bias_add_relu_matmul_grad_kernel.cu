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

class MatmulGradKernelState final : public user_op::OpKernelState {
 public:
  MatmulGradKernelState() {
    OF_CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    OF_CUBLAS_CHECK(cublasLtCreate(&cublas_lt_handle_));
    OF_CUDA_CHECK(cudaMalloc(&workspace_, 8 * 1024 * 1024));
  }
  ~MatmulGradKernelState() {
    OF_CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
    OF_CUBLAS_CHECK(cublasLtDestroy(cublas_lt_handle_));
    OF_CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
    OF_CUDA_CHECK(cudaFree(workspace_));
  }
  cudaStream_t cuda_stream() const { return cuda_stream_; }
  cublasLtHandle_t cublas_lt_handle() const { return cublas_lt_handle_; }
  size_t cublas_workspace_size() const { return 8 * 1024 * 1024; }
  void* cublas_workspace() const { return workspace_; }

 private:
  cudaStream_t cuda_stream_{};
  cublasLtHandle_t cublas_lt_handle_{};
  void* workspace_{};
};

template<typename T>
class CublasBiasAddReluMatmulGradKernel final : public user_op::OpKernel,
                                                public user_op::CudaGraphSupport {
 public:
  CublasBiasAddReluMatmulGradKernel() {
    OF_CUDA_CHECK(cudaEventCreate(&main_stream_event));
    OF_CUDA_CHECK(cudaEventCreate(&async_matmul_grad_event));
  };
  ~CublasBiasAddReluMatmulGradKernel() override {
    OF_CUDA_CHECK(cudaEventDestroy(main_stream_event));
    OF_CUDA_CHECK(cudaEventDestroy(async_matmul_grad_event));
  };

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<MatmulGradKernelState>();
  }

 private:
  cudaEvent_t main_stream_event;
  cudaEvent_t async_matmul_grad_event;

  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* aux = ctx->Tensor4ArgNameAndIndex("aux", 0);
    const user_op::Tensor* hidden = ctx->Tensor4ArgNameAndIndex("hidden", 0);

    user_op::Tensor* d_bias = ctx->Tensor4ArgNameAndIndex("d_bias", 0);
    user_op::Tensor* d_grad = ctx->Tensor4ArgNameAndIndex("d_grad", 0);
    user_op::Tensor* d_weight = ctx->Tensor4ArgNameAndIndex("d_weight", 0);

    const auto* matmul_grad_cache =
        CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    auto* kernel_state = dynamic_cast<MatmulGradKernelState*>(state);

    const DataType data_type = dy->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

    double alpha = ctx->Attr<double>("alpha");
    auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    double beta = 0.0;
    auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    // currently only support 2D matmul.
    DimVector dy_shape(2);
    dy->shape().ToDimVector(&dy_shape);
    DimVector weight_shape(2);
    weight->shape().ToDimVector(&weight_shape);
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DRELU_BGRAD;

    InferMatmulCublasMNK(dy_shape, weight_shape,
                         /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                         /*transpose_b=*/ep::primitive::BlasTransposeType::N, &cublas_m, &cublas_n,
                         &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);

    SetCublasAttr(matmul_grad_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/true,
                  /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                  /*transpose_b=*/ep::primitive::BlasTransposeType::N, epilogue, d_bias->dptr(),
                  aux->dptr(), cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc, 1024 * 1024 * 8);
    /*
    a = dy, b = weight
    cublas_a=weight, cublas_b=dy
    */
    OF_CUDA_CHECK(cudaEventRecord(main_stream_event, cuda_stream->cuda_stream()));
    OF_CUBLAS_CHECK(
        cublasLtMatmul(cuda_stream->cublas_lt_handle(), matmul_grad_cache->operation_desc,
                       &sp_alpha, weight->dptr(), matmul_grad_cache->cublas_a_desc, dy->dptr(),
                       matmul_grad_cache->cublas_b_desc, &sp_beta, d_grad->mut_dptr(),
                       matmul_grad_cache->cublas_c_desc, d_grad->mut_dptr(),
                       matmul_grad_cache->cublas_c_desc, nullptr, cuda_stream->cublas_workspace(),
                       cuda_stream->cublas_workspace_size(), cuda_stream->cuda_stream()));

    alpha = 1.0;
    sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    beta = 0.0;
    sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    // currently only support 2D matmul.
    DimVector hidden_shape(2);
    hidden->shape().ToDimVector(&hidden_shape);
    epilogue = CUBLASLT_EPILOGUE_DEFAULT;

    InferMatmulCublasMNK(dy_shape, hidden_shape,
                         /*transpose_a=*/ep::primitive::BlasTransposeType::T,
                         /*transpose_b=*/ep::primitive::BlasTransposeType::N, &cublas_m, &cublas_n,
                         &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);

    SetCublasAttr(matmul_grad_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/false,
                  /*transpose_a=*/ep::primitive::BlasTransposeType::T,
                  /*transpose_b=*/ep::primitive::BlasTransposeType::N, epilogue, nullptr, nullptr,
                  cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc, 8 * 1024 * 1024);
    OF_CUDA_CHECK(cudaStreamWaitEvent(kernel_state->cuda_stream(), main_stream_event));
    OF_CUBLAS_CHECK(
        cublasLtMatmul(kernel_state->cublas_lt_handle(), matmul_grad_cache->operation_desc,
                       &sp_alpha, hidden->dptr(), matmul_grad_cache->cublas_a_desc, dy->dptr(),
                       matmul_grad_cache->cublas_b_desc, &sp_beta, d_weight->mut_dptr(),
                       matmul_grad_cache->cublas_c_desc, d_weight->mut_dptr(),
                       matmul_grad_cache->cublas_c_desc, nullptr, kernel_state->cublas_workspace(),
                       kernel_state->cublas_workspace_size(), kernel_state->cuda_stream()));
    OF_CUDA_CHECK(cudaEventRecord(async_matmul_grad_event, kernel_state->cuda_stream()));
    OF_CUDA_CHECK(cudaStreamWaitEvent(cuda_stream->cuda_stream(), async_matmul_grad_event));
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
