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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cuda.h>

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

void InferMatmulCublasMNK(const ShapeView& a_shape, const ShapeView& b_shape, 
                          ep::primitive::BlasTransposeType transpose_a,
                          ep::primitive::BlasTransposeType transpose_b, 
                          size_t* cublas_m, size_t* cublas_n, size_t* cublas_k, 
                          int64_t* cublas_lda, int64_t* cublas_ldb, int64_t* cublas_ldc) {
  const int64_t num_a_axes = a_shape.NumAxes();
  CHECK_GE(num_a_axes, 2);
  const int64_t num_b_axes = b_shape.NumAxes();
  CHECK_GE(num_b_axes, 2);
  size_t m = 0, n = 0, k = 0; 
  if (transpose_a == ep::primitive::BlasTransposeType::N) {
    m = a_shape.At(num_a_axes - 2);
    k = a_shape.At(num_a_axes - 1);
    *cublas_ldb = k;
  } else if (transpose_a == ep::primitive::BlasTransposeType::T) {
    m = a_shape.At(num_a_axes - 1);
    k = a_shape.At(num_a_axes - 2);
    *cublas_ldb = m;
  } else {
    UNIMPLEMENTED();
  }
  if (transpose_b == ep::primitive::BlasTransposeType::N) {
    CHECK_EQ(b_shape.At(num_b_axes - 2), k);
    n = b_shape.At(num_b_axes - 1);
    *cublas_lda = n;
  } else if (transpose_b == ep::primitive::BlasTransposeType::T) {
    CHECK_EQ(b_shape.At(num_b_axes - 1), k);
    n = b_shape.At(num_b_axes - 2);
    *cublas_lda = k;
  } else {
    UNIMPLEMENTED();
  }
  *cublas_m = n; 
  *cublas_n = m; 
  *cublas_k = k; 
  *cublas_ldc = n;
}

class FusedMatmulBiasAddReluKernelCache final : public user_op::OpKernelCache {
 public:
  FusedMatmulBiasAddReluKernelCache() {
// Just for init.
    OF_CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation_desc1, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_a1_desc, CUDA_R_32F, 1, 1, 1));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_b1_desc, CUDA_R_32F, 1, 1, 1));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_c1_desc, CUDA_R_32F, 1, 1, 1));
    
  }
  ~FusedMatmulBiasAddReluKernelCache() override {
    OF_CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation_desc1));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_a1_desc));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_b1_desc));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_c1_desc));
  }
  cublasLtMatmulDesc_t operation_desc1;
  cublasLtMatrixLayout_t cublas_a1_desc;
  cublasLtMatrixLayout_t cublas_b1_desc;
  cublasLtMatrixLayout_t cublas_c1_desc;
};

std::shared_ptr<FusedMatmulBiasAddReluKernelCache> CreateFusedMatmulBiasAddReluKernelCache() {
  std::shared_ptr<FusedMatmulBiasAddReluKernelCache> cache(new FusedMatmulBiasAddReluKernelCache());
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

void SetCublasEpilogue(const FusedMatmulBiasAddReluKernelCache* matmul_cache, 
                       cublasLtEpilogue_t epilogue, 
                       const void* bias_ptr, 
                       const void* aux_ptr){
  // if(epilogue == CUBLASLT_EPILOGUE_RELU_BIAS || epilogue == CUBLASLT_EPILOGUE_RELU_AUX_BIAS){
  //   // Set bias ptr
  //   OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc1,
  //     CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
  //     sizeof(bias_ptr)));
  // }
  if(epilogue == CUBLASLT_EPILOGUE_RELU_BIAS || epilogue == CUBLASLT_EPILOGUE_BIAS){
    // Set epilogue
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_cache->operation_desc1, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    // Set bias ptr
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc1,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
      sizeof(bias_ptr)));
  }
  // // TODO: GELU_AUX_BIAS
  // if(epilogue == CUBLASLT_EPILOGUE_RELU_AUX_BIAS){
  //   // Set aux ptr for backward. 
  //   OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc1,
  //     CUBLASLT_MATMUL_DESC_AUX_POINTER, &aux_ptr,
  //     sizeof(aux_ptr)));
  // }
}

void SetCublasAttr(const FusedMatmulBiasAddReluKernelCache* matmul_cache, 
                   const cublasComputeType_t cublas_compute_dtype, 
                   const cudaDataType_t cuda_data_type, 
                   cublasLtEpilogue_t epilogue, 
                   const void* bias_ptr, 
                   const void* aux_ptr, 
                   size_t cublas_m, 
                   size_t cublas_n, 
                   size_t cublas_k, 
                   int64_t cublas_lda, 
                   int64_t cublas_ldb, 
                   int64_t cublas_ldc
                   ){
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
    matmul_cache->operation_desc1, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, &cublas_compute_dtype,
    sizeof(cublas_compute_dtype)));

  // For best performance when using the bias vector, specify beta == 0 and
  // CUBLASLT_POINTER_MODE_HOST.(from
  // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtPointerMode_t)
  cublasLtPointerMode_t mode = CUBLASLT_POINTER_MODE_HOST;
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_cache->operation_desc1, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)));
  
  // transpose_a = False, transpose_b = True. But in cublas is reversed. 
  const cublasOperation_t cublas_trans_a = CUBLAS_OP_T;
  const cublasOperation_t cublas_trans_b = CUBLAS_OP_N;
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc1,
                                                CUBLASLT_MATMUL_DESC_TRANSA, &cublas_trans_a,
                                                sizeof(cublas_trans_a)));
  OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc1,
                                                CUBLASLT_MATMUL_DESC_TRANSB, &cublas_trans_b,
                                                sizeof(cublas_trans_b)));
  
  // Set epilogue
  // OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
  //     matmul_cache->operation_desc1, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  SetCublasEpilogue(matmul_cache, epilogue, bias_ptr, aux_ptr);

  // // Set bias ptr
  // OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc1,
  //                                                CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
  //                                                sizeof(bias_ptr)));
  
  // Set matrix layout
  SetCublasMatrixLayout(matmul_cache->cublas_a1_desc, cuda_data_type, cublas_trans_a, cublas_m,
                        cublas_k, cublas_lda);
  SetCublasMatrixLayout(matmul_cache->cublas_b1_desc, cuda_data_type, cublas_trans_b, cublas_k,
                        cublas_n, cublas_ldb);
  SetCublasMatrixLayout(matmul_cache->cublas_c1_desc, cuda_data_type, CUBLAS_OP_N, cublas_m,
                        cublas_n, cublas_ldc);
}

}  // namespace

template<typename T>
class FusedMatmulBiasAddReluKernel final : public user_op::OpKernel {
 public:
  FusedMatmulBiasAddReluKernel() = default;
  ~FusedMatmulBiasAddReluKernel() override = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateFusedMatmulBiasAddReluKernelCache();
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    /*
    Fused Dense+Activation+Dense Layer. 
    A: (m, k)
    B: (n, k) need transpose
    C: (j, n) need transpose
    tmp: A matmul B(transpose), its shape is (m, n)
    out: tmp matmul C(transpose), its shape is (m, j)
    */
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    const user_op::Tensor* c = ctx->Tensor4ArgNameAndIndex("c", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    const auto* matmul_cache =
        CHECK_NOTNULL(dynamic_cast<const FusedMatmulBiasAddReluKernelCache*>(cache));

    const user_op::Tensor* bias1 = ctx->Tensor4ArgNameAndIndex("bias1", 0);
    const user_op::Tensor* bias2 = ctx->Tensor4ArgNameAndIndex("bias2", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* aux = ctx->Tensor4ArgNameAndIndex("aux", 0);
    
    const DataType data_type = out->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    
    const user_op::Tensor* cublas_a = b;
    const user_op::Tensor* cublas_b = a;
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0; 
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0; 
    InferMatmulCublasMNK(a->shape(), b->shape(), 
                         /*transpose_a=*/ep::primitive::BlasTransposeType::N, 
                         /*transpose_b=*/ep::primitive::BlasTransposeType::T, 
                         &cublas_m, &cublas_n, &cublas_k, 
                         &cublas_lda, &cublas_ldb, &cublas_ldc); 
    const double alpha1 = ctx->Attr<double>("alpha1");
    const auto sp_alpha1 = GetCublasScalarParameter(alpha1, cublas_compute_dtype);
    const double beta1 = 0.0;
    const auto sp_beta1 = GetCublasScalarParameter(beta1, cublas_compute_dtype);
    // First matmul + bias + relu. 
    // cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_AUX_BIAS;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    SetCublasAttr(matmul_cache, 
                  cublas_compute_dtype, 
                  cuda_data_type, 
                  epilogue, 
                  bias1->dptr(), 
                  aux->dptr(), 
                  cublas_m, 
                  cublas_n, 
                  cublas_k, 
                  cublas_lda, 
                  cublas_ldb, 
                  cublas_ldc); 
    OF_CUBLAS_CHECK(cublasLtMatmul(
        cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc1, &sp_alpha1, cublas_a->dptr(),
        matmul_cache->cublas_a1_desc, cublas_b->dptr(), matmul_cache->cublas_b1_desc, &sp_beta1,
        tmp_buffer->mut_dptr(), matmul_cache->cublas_c1_desc, tmp_buffer->mut_dptr(), matmul_cache->cublas_c1_desc,
        nullptr, cuda_stream->cublas_workspace(), cuda_stream->cublas_workspace_size(),
        cuda_stream->cuda_stream()));
    
    // A shape is (m, k), B shape is (n, k) which need transpose, so the tmp_out shape is (m, n)
    const ShapeView tmp_shape({a->shape().At(a->shape().NumAxes()-2), b->shape().At(b->shape().NumAxes()-2)}); 
    InferMatmulCublasMNK(tmp_shape, c->shape(), 
                         /*transpose_a=*/ep::primitive::BlasTransposeType::N, 
                         /*transpose_b=*/ep::primitive::BlasTransposeType::T, 
                         &cublas_m, &cublas_n, &cublas_k, 
                         &cublas_lda, &cublas_ldb, &cublas_ldc); 

    cublas_a = c;
    cublas_b = tmp_buffer;
    const double alpha2 = ctx->Attr<double>("alpha2");
    const auto sp_alpha2 = GetCublasScalarParameter(alpha2, cublas_compute_dtype);
    const double beta2 = 0.0;
    const auto sp_beta2 = GetCublasScalarParameter(beta2, cublas_compute_dtype);
    // Second matmul + bias
    epilogue = CUBLASLT_EPILOGUE_BIAS;
    SetCublasAttr(matmul_cache, 
                  cublas_compute_dtype, 
                  cuda_data_type, 
                  epilogue, 
                  bias2->dptr(),
                  nullptr,  
                  cublas_m, 
                  cublas_n, 
                  cublas_k, 
                  cublas_lda, 
                  cublas_ldb, 
                  cublas_ldc); 
    OF_CUBLAS_CHECK(cublasLtMatmul(
        cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc1, &sp_alpha2, cublas_a->dptr(),
        matmul_cache->cublas_a1_desc, cublas_b->dptr(), matmul_cache->cublas_b1_desc, &sp_beta2,
        out->mut_dptr(), matmul_cache->cublas_c1_desc, out->mut_dptr(), matmul_cache->cublas_c1_desc,
        nullptr, cuda_stream->cublas_workspace(), cuda_stream->cublas_workspace_size(),
        cuda_stream->cuda_stream()));
  }
};

#define REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(cpp_type, data_type)  \
  REGISTER_USER_KERNEL("fused_matmul_bias_add_relu")                   \
      .SetCreateFn<FusedMatmulBiasAddReluKernel<cpp_type>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == data_type)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
      const Shape& a_shape = ctx->InputShape("a", 0);                                \
      const int64_t a_num_axes = a_shape.NumAxes();                                 \
      const Shape& b_shape = ctx->InputShape("b", 0);                                 \
      const int64_t b_num_axes = b_shape.NumAxes();                                 \
      const int64_t tmp_buffer_size =                                                 \
        GetCudaAlignedSize(a_shape.Count(0, a_num_axes-1)* b_shape.At(b_num_axes-2) * sizeof(cpp_type));\
      return tmp_buffer_size;                                                         \
    });

REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(double, DataType::kDouble);
REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(float, DataType::kFloat);
REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(half, DataType::kFloat16);
#if CUDA_VERSION >= 11000
REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(nv_bfloat16, DataType::kBFloat16);
#endif

}  // namespace oneflow
