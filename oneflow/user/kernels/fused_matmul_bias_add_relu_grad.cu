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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

Optional<cudaDataType_t> OptCudaDataType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUDA_R_32F;
    case kDouble: return CUDA_R_64F;
    case kFloat16: return CUDA_R_16F;
#if CUDA_VERSION >= 11000
    case kBFloat16: return CUDA_R_16BF;
#endif  // CUDA_VERSION >= 11000
    default: return NullOpt;
  }
}

cudaDataType_t GetCudaDataType(DataType data_type) {
  auto cuda_data_type = OptCudaDataType(data_type);
  CHECK(cuda_data_type.has_value());
  return cuda_data_type.value_or(CUDA_R_32F);
}

cublasComputeType_t GetComputeType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUBLAS_COMPUTE_32F;
    case kDouble: return CUBLAS_COMPUTE_64F;
    case kFloat16: return CUBLAS_COMPUTE_32F;
    case kBFloat16: return CUBLAS_COMPUTE_32F;
    default: UNIMPLEMENTED(); return CUBLAS_COMPUTE_32F;
  }
}

union CublasScalarParameter {
  double d;
  float s;
};

CublasScalarParameter GetCublasScalarParameter(Scalar scalar, cublasComputeType_t compute_type) {
  CublasScalarParameter sp{};
  if (compute_type == CUBLAS_COMPUTE_64F) {
    sp.d = scalar.Value<double>();
  } else if (compute_type == CUBLAS_COMPUTE_32F) {
    sp.s = scalar.Value<float>();
  } else {
    UNIMPLEMENTED();
  }
  return sp;
}

void InferMatmulCublasMNK(const DimVector& a_shape, const DimVector& b_shape, 
                          ep::primitive::BlasTransposeType transpose_a,
                          ep::primitive::BlasTransposeType transpose_b, 
                          size_t* cublas_m, size_t* cublas_n, size_t* cublas_k, 
                          int64_t* cublas_lda, int64_t* cublas_ldb, int64_t* cublas_ldc) {
    const int64_t num_a_axes = a_shape.size();
    CHECK_GE(num_a_axes, 2);
    const int64_t num_b_axes = b_shape.size();
    CHECK_GE(num_b_axes, 2);
    size_t m = 0, n = 0, k = 0; 
    if (transpose_a == ep::primitive::BlasTransposeType::N) {
      m = a_shape.at(num_a_axes - 2);
      k = a_shape.at(num_a_axes - 1);
      *cublas_ldb = k;
    } else if (transpose_a == ep::primitive::BlasTransposeType::T) {
      m = a_shape.at(num_a_axes - 1);
      k = a_shape.at(num_a_axes - 2);
      *cublas_ldb = m;
    } else {
      UNIMPLEMENTED();
    }
    if (transpose_b == ep::primitive::BlasTransposeType::N) {
      CHECK_EQ(b_shape.at(num_b_axes - 2), k);
      n = b_shape.at(num_b_axes - 1);
      *cublas_lda = n;
    } else if (transpose_b == ep::primitive::BlasTransposeType::T) {
      CHECK_EQ(b_shape.at(num_b_axes - 1), k);
      n = b_shape.at(num_b_axes - 2);
      *cublas_lda = k;
    } else {
      UNIMPLEMENTED();
    }
    *cublas_m = n; 
    *cublas_n = m; 
    *cublas_k = k; 
    *cublas_ldc = n;
  }


class FusedMatmulBiasAddReluGradKernelCache final : public user_op::OpKernelCache {
  public:
  FusedMatmulBiasAddReluGradKernelCache() {
  // Just for init.
  OF_CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_a_desc, CUDA_R_32F, 1, 1, 1));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_b_desc, CUDA_R_32F, 1, 1, 1));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_c_desc, CUDA_R_32F, 1, 1, 1));
  }
  ~FusedMatmulBiasAddReluGradKernelCache() override {
    OF_CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation_desc));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_a_desc));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_b_desc));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_c_desc));
  }
  cublasLtMatmulDesc_t operation_desc;
  cublasLtMatrixLayout_t cublas_a_desc;
  cublasLtMatrixLayout_t cublas_b_desc;
  cublasLtMatrixLayout_t cublas_c_desc;
};

std::shared_ptr<FusedMatmulBiasAddReluGradKernelCache> CreateFusedMatmulBiasAddReluGradKernelCache() {
  std::shared_ptr<FusedMatmulBiasAddReluGradKernelCache> cache(new FusedMatmulBiasAddReluGradKernelCache());
  return cache;
}

void SetCublasMatrixLayout(cublasLtMatrixLayout_t layout_desc, cudaDataType_t cuda_data_type,
                            cublasOperation_t cublas_trans, const size_t cublas_m1,
                            const size_t cublas_n1, int64_t cublas_ld) {
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layout_desc, CUBLASLT_MATRIX_LAYOUT_TYPE,
                                                    &cuda_data_type, sizeof(cuda_data_type)));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_desc, CUBLASLT_MATRIX_LAYOUT_ROWS, cublas_trans == CUBLAS_OP_N ? &cublas_m1 : &cublas_n1,
      sizeof(cublas_m1)));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_desc, CUBLASLT_MATRIX_LAYOUT_COLS, cublas_trans == CUBLAS_OP_N ? &cublas_n1 : &cublas_m1,
      sizeof(cublas_m1)));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layout_desc, CUBLASLT_MATRIX_LAYOUT_LD,
                                                    &cublas_ld, sizeof(cublas_ld)));
}

void SetCublasEpilogue(const FusedMatmulBiasAddReluGradKernelCache* matmul_grad_cache, 
                        cublasLtEpilogue_t epilogue, 
                        const void* d_bias_ptr, 
                        const void* aux_ptr){
  // TODO: Support GELU_AUX_BIAS. 
  if(epilogue == CUBLASLT_EPILOGUE_DRELU_BGRAD){
    // Set epilogue
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_grad_cache->operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    // Set bias ptr
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_grad_cache->operation_desc,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias_ptr,
      sizeof(d_bias_ptr)));
    // Set aux ptr
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_grad_cache->operation_desc,
      CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux_ptr,
      sizeof(aux_ptr)));
  }

}

void SetCublasAttr(const FusedMatmulBiasAddReluGradKernelCache* matmul_grad_cache, 
                    const cublasComputeType_t cublas_compute_dtype, 
                    const cudaDataType_t cuda_data_type, 
                    ep::primitive::BlasTransposeType transpose_a, 
                    ep::primitive::BlasTransposeType transpose_b,
                    cublasLtEpilogue_t epilogue, 
                    const void* d_bias_ptr, 
                    const void* aux_ptr, 
                    size_t cublas_m, 
                    size_t cublas_n, 
                    size_t cublas_k, 
                    int64_t cublas_lda, 
                    int64_t cublas_ldb, 
                    int64_t cublas_ldc
                    ){
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
    matmul_grad_cache->operation_desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, &cublas_compute_dtype,
    sizeof(cublas_compute_dtype)));

  // For best performance when using the bias vector, specify beta == 0 and
  // CUBLASLT_POINTER_MODE_HOST.(from
  // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtPointerMode_t)
  cublasLtPointerMode_t mode = CUBLASLT_POINTER_MODE_HOST;
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_grad_cache->operation_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)));
  
  // transpose_a = False, transpose_b = True. But in cublas is reversed. 
  const cublasOperation_t cublas_trans_a = transpose_b == ep::primitive::BlasTransposeType::T ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t cublas_trans_b = transpose_a == ep::primitive::BlasTransposeType::T ? CUBLAS_OP_T : CUBLAS_OP_N;
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_grad_cache->operation_desc,
                                                CUBLASLT_MATMUL_DESC_TRANSA, &cublas_trans_a,
                                                sizeof(cublas_trans_a)));
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_grad_cache->operation_desc,
                                                CUBLASLT_MATMUL_DESC_TRANSB, &cublas_trans_b,
                                                sizeof(cublas_trans_b)));
  
  // Set epilogue
  SetCublasEpilogue(matmul_grad_cache, epilogue, d_bias_ptr, aux_ptr);
  /*
  Set AUX pointer LD
  If is used for CUBLASLT_EPILOGUE_DRELU_BGRAD, the AUX_LD need to align 128bit.
  If is used for CUBLASLT_EPILOGUE_DGELU_BGRAD, the AUX_LD need to align 8.
  For more details you can refer to CUBLAS docs: https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulDescAttributes_t `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD`. 
  */
  long aux_ld = cublas_ldc; 
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_grad_cache->operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, 
    &aux_ld, sizeof(aux_ld)));
    
  // Set matrix layout
  SetCublasMatrixLayout(matmul_grad_cache->cublas_a_desc, cuda_data_type, cublas_trans_a, cublas_m,
                        cublas_k, cublas_lda);
  SetCublasMatrixLayout(matmul_grad_cache->cublas_b_desc, cuda_data_type, cublas_trans_b, cublas_k,
                        cublas_n, cublas_ldb);
  SetCublasMatrixLayout(matmul_grad_cache->cublas_c_desc, cuda_data_type, CUBLAS_OP_N, cublas_m,
                        cublas_n, cublas_ldc);
}

}  // namespace


template<typename T>
class FusedMatmulBiasAddReluGradKernel final : public user_op::OpKernel {
 public:
  FusedMatmulBiasAddReluGradKernel() = default;
  ~FusedMatmulBiasAddReluGradKernel() override = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateFusedMatmulBiasAddReluGradKernelCache();
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
    CHECK_NOTNULL(dynamic_cast<const FusedMatmulBiasAddReluGradKernelCache*>(cache));
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
    DimVector weight_shape(2); 
    weight->shape().ToDimVector(&weight_shape); 
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DRELU_BGRAD;
    
    InferMatmulCublasMNK(dy_shape, weight_shape, 
                        /*transpose_a=*/ep::primitive::BlasTransposeType::N, 
                        /*transpose_b=*/ep::primitive::BlasTransposeType::N, 
                        &cublas_m, &cublas_n, &cublas_k, 
                        &cublas_lda, &cublas_ldb, &cublas_ldc);
    
    SetCublasAttr(matmul_grad_cache, 
                  cublas_compute_dtype, 
                  cuda_data_type, 
                  /*transpose_a=*/ep::primitive::BlasTransposeType::N, 
                  /*transpose_b=*/ep::primitive::BlasTransposeType::N, 
                  epilogue, 
                  d_bias->dptr(), 
                  aux->dptr(), 
                  cublas_m, 
                  cublas_n, 
                  cublas_k, 
                  cublas_lda, 
                  cublas_ldb, 
                  cublas_ldc); 
    /*
    a = dy, b = weight
    cublas_a=weight, cublas_b=dy
    */
    OF_CUBLAS_CHECK(cublasLtMatmul(
        cuda_stream->cublas_lt_handle(), 
        matmul_grad_cache->operation_desc, &sp_alpha, 
        weight->dptr(),
        matmul_grad_cache->cublas_a_desc,  
        dy->dptr(),
        matmul_grad_cache->cublas_b_desc, 
        &sp_beta,
        d_grad->mut_dptr(), 
        matmul_grad_cache->cublas_c_desc, 
        d_grad->mut_dptr(), 
        matmul_grad_cache->cublas_c_desc,
        nullptr, cuda_stream->cublas_workspace(), 
        cuda_stream->cublas_workspace_size(),
        cuda_stream->cuda_stream()));

  };
};

#define REGISTER_FUSED_MATMUL_BIAS_ADD_GELU_GRAD_KERNEL(dtype)                \
  REGISTER_USER_KERNEL("fused_matmul_bias_add_relu_backward")                     \
      .SetCreateFn<FusedMatmulBiasAddReluGradKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("weight", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_MATMUL_BIAS_ADD_GELU_GRAD_KERNEL(float)
REGISTER_FUSED_MATMUL_BIAS_ADD_GELU_GRAD_KERNEL(double)
// REGISTER_FUSED_MATMUL_BIAS_ADD_GELU_GRAD_KERNEL(half)

}  // namespace oneflow
