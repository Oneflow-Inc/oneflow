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

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

template<typename Context>
ep::primitive::BlasTransposeType GetBlasTransposeType(Context* ctx, const std::string& attr) {
  return GetBlasTransposeType(ctx->template Attr<bool>(attr));
}

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
#if CUDA_VERSION >= 11000
    case kBFloat16: return CUBLAS_COMPUTE_32F;
#endif  // CUDA_VERSION >= 11000
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

void InferMatmulMNK(const ShapeView& a_shape, const ShapeView& b_shape, const ShapeView& c_shape,
                    ep::primitive::BlasTransposeType transpose_a,
                    ep::primitive::BlasTransposeType transpose_b, size_t* m, size_t* n, size_t* k) {
  const int64_t num_a_axes = a_shape.NumAxes();
  CHECK_GE(num_a_axes, 2);
  const int64_t num_b_axes = b_shape.NumAxes();
  CHECK_GE(num_b_axes, 2);
  const int64_t num_c_axes = c_shape.NumAxes();
  CHECK_GE(num_c_axes, 2);
  if (transpose_a == ep::primitive::BlasTransposeType::N) {
    *m = a_shape.At(num_a_axes - 2);
    *k = a_shape.At(num_a_axes - 1);
  } else if (transpose_a == ep::primitive::BlasTransposeType::T) {
    *m = a_shape.At(num_a_axes - 1);
    *k = a_shape.At(num_a_axes - 2);
  } else {
    UNIMPLEMENTED();
  }
  if (transpose_b == ep::primitive::BlasTransposeType::N) {
    CHECK_EQ(b_shape.At(num_b_axes - 2), *k);
    *n = b_shape.At(num_b_axes - 1);
  } else if (transpose_b == ep::primitive::BlasTransposeType::T) {
    CHECK_EQ(b_shape.At(num_b_axes - 1), *k);
    *n = b_shape.At(num_b_axes - 2);
  } else {
    UNIMPLEMENTED();
  }
  CHECK_EQ(c_shape.At(num_c_axes - 2), *m);
  CHECK_EQ(c_shape.At(num_c_axes - 1), *n);
}

class FusedMatmulBiasAddReluKernelCache final : public user_op::OpKernelCache {
 public:
  FusedMatmulBiasAddReluKernelCache() {
    // Just for init.
    OF_CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_a_desc, CUDA_R_32F, 1, 1, 1));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_b_desc, CUDA_R_32F, 1, 1, 1));
    OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_c_desc, CUDA_R_32F, 1, 1, 1));
  }
  ~FusedMatmulBiasAddReluKernelCache() override {
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

std::shared_ptr<FusedMatmulBiasAddReluKernelCache> CreateFusedMatmulBiasAddReluKernelCache() {
  std::shared_ptr<FusedMatmulBiasAddReluKernelCache> cache(new FusedMatmulBiasAddReluKernelCache());
  return cache;
}

void SetCublasMatrixLayout(cublasLtMatrixLayout_t layout_desc, cudaDataType_t cuda_data_type,
                           cublasOperation_t cublas_trans, const size_t cublas_m,
                           const size_t cublas_n, int64_t cublas_ld) {
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layout_desc, CUBLASLT_MATRIX_LAYOUT_TYPE,
                                                   &cuda_data_type, sizeof(cuda_data_type)));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_desc, CUBLASLT_MATRIX_LAYOUT_ROWS, cublas_trans == CUBLAS_OP_N ? &cublas_m : &cublas_n,
      sizeof(cublas_m)));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_desc, CUBLASLT_MATRIX_LAYOUT_COLS, cublas_trans == CUBLAS_OP_N ? &cublas_n : &cublas_m,
      sizeof(cublas_m)));
  OF_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layout_desc, CUBLASLT_MATRIX_LAYOUT_LD,
                                                   &cublas_ld, sizeof(cublas_ld)));
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
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    const user_op::Tensor* cublas_a = b;
    const user_op::Tensor* cublas_b = a;

    const auto* matmul_cache =
        CHECK_NOTNULL(dynamic_cast<const FusedMatmulBiasAddReluKernelCache*>(cache));

    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType data_type = out->data_type();

    const auto GetCublasOperation = [](ep::primitive::BlasTransposeType transpose_type) {
      if (transpose_type == ep::primitive::BlasTransposeType::N) {
        return CUBLAS_OP_N;
      } else if (transpose_type == ep::primitive::BlasTransposeType::T) {
        return CUBLAS_OP_T;
      } else {
        UNIMPLEMENTED();
        return CUBLAS_OP_N;
      }
    };

    const auto trans_a = GetBlasTransposeType(ctx, "transpose_a");
    const auto trans_b = GetBlasTransposeType(ctx, "transpose_b");

    size_t m = 0, n = 0, k = 0;
    InferMatmulMNK(a->shape(), b->shape(), out->shape(), trans_a, trans_b, &m, &n, &k);

    /*
    Matmul: A(m, k) x B(k, n) = C(m, n), it follows the row major.
    In cublas, it use column major to compute, Bt(n, k) x At(k, m) = Ct(n, m).
    And Ct matrix follows the column major is equal to C(m, n) which follows the row major.
    */
    const size_t cublas_m = n;
    const size_t cublas_n = m;
    const size_t cublas_k = k;

    const cublasOperation_t cublas_trans_a = GetCublasOperation(trans_b);
    const cublasOperation_t cublas_trans_b = GetCublasOperation(trans_a);

    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);

    const double alpha = ctx->Attr<double>("alpha");
    const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);

    const double beta = 0.0;
    const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    int64_t cublas_lda = 0;
    if (trans_b == ep::primitive::BlasTransposeType::N) {
      cublas_lda = n;
    } else if (trans_b == ep::primitive::BlasTransposeType::T) {
      cublas_lda = k;
    } else {
      UNIMPLEMENTED();
    }

    int64_t cublas_ldb = 0;
    if (trans_a == ep::primitive::BlasTransposeType::N) {
      cublas_ldb = k;
    } else if (trans_a == ep::primitive::BlasTransposeType::T) {
      cublas_ldb = m;
    } else {
      UNIMPLEMENTED();
    }
    const int64_t cublas_ldc = n;

    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        matmul_cache->operation_desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, &cublas_compute_dtype,
        sizeof(cublas_compute_dtype)));
    // For best performance when using the bias vector, specify beta == 0 and
    // CUBLASLT_POINTER_MODE_HOST.(from
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtPointerMode_t)
    cublasLtPointerMode_t mode = CUBLASLT_POINTER_MODE_HOST;
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        matmul_cache->operation_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)));
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc,
                                                   CUBLASLT_MATMUL_DESC_TRANSA, &cublas_trans_a,
                                                   sizeof(cublas_trans_a)));
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc,
                                                   CUBLASLT_MATMUL_DESC_TRANSB, &cublas_trans_b,
                                                   sizeof(cublas_trans_b)));

    // Set as matmul + bias_add + relu.
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        matmul_cache->operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // Set bias ptr
    const T* bias_ptr = reinterpret_cast<const T*>(bias->dptr());
    OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_cache->operation_desc,
                                                   CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
                                                   sizeof(bias_ptr)));

    SetCublasMatrixLayout(matmul_cache->cublas_a_desc, cuda_data_type, cublas_trans_a, cublas_m,
                          cublas_k, cublas_lda);
    SetCublasMatrixLayout(matmul_cache->cublas_b_desc, cuda_data_type, cublas_trans_b, cublas_k,
                          cublas_n, cublas_ldb);
    SetCublasMatrixLayout(matmul_cache->cublas_c_desc, cuda_data_type, CUBLAS_OP_N, cublas_m,
                          cublas_n, cublas_ldc);

    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    OF_CUBLAS_CHECK(cublasLtMatmul(
        cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &sp_alpha, cublas_a->dptr(),
        matmul_cache->cublas_a_desc, cublas_b->dptr(), matmul_cache->cublas_b_desc, &sp_beta,
        out->mut_dptr(), matmul_cache->cublas_c_desc, out->mut_dptr(), matmul_cache->cublas_c_desc,
        nullptr, cuda_stream->cublas_workspace(), cuda_stream->cublas_workspace_size(),
        cuda_stream->cuda_stream()));
  }
};

#define REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(cpp_type, data_type)  \
  REGISTER_USER_KERNEL("fused_matmul_bias_add_relu")                   \
      .SetCreateFn<FusedMatmulBiasAddReluKernel<cpp_type>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == data_type));

REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(double, DataType::kDouble);
REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(float, DataType::kFloat);
REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(half, DataType::kFloat16);
#if CUDA_VERSION >= 11000
// REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(nv_bfloat16, DataType::kBFloat16);
#endif
}  // namespace oneflow
