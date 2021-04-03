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
#include <cusparse.h>

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
    }                                                                          \
  }

namespace oneflow {
namespace {
class SpmmCOOGpuFloatKernel final : public user_op::OpKernel {
 public:
  SpmmCOOGpuFloatKernel() = default;
  ~SpmmCOOGpuFloatKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const int64_t A_num_rows = ctx->Attr<int64_t>("a_rows");
    const int64_t A_num_cols = ctx->Attr<int64_t>("a_cols");

    const user_op::Tensor *a_cooRowInd = ctx->Tensor4ArgNameAndIndex("a_cooRowInd", 0);
    const user_op::Tensor *a_cooColInd = ctx->Tensor4ArgNameAndIndex("a_cooColInd", 0);
    const user_op::Tensor *a_cooValues = ctx->Tensor4ArgNameAndIndex("a_cooValues", 0);
    const user_op::Tensor *b = ctx->Tensor4ArgNameAndIndex("b", 0);

    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t *a_cooRowInd_ptr = a_cooRowInd->dptr<int64_t>();
    const int64_t *a_cooColInd_ptr = a_cooColInd->dptr<int64_t>();
    const float *a_cooValues_ptr = a_cooValues->dptr<float>();
    const float *b_ptr = b->dptr<float>();

    float *out_ptr = out_tensor->mut_dptr<float>();

    int A_nnz = a_cooRowInd->shape().At(0);
    int B_num_rows = b->shape().At(0);
    int B_num_cols = b->shape().At(1);

    int ldb = B_num_cols;
    int ldc = B_num_cols;
    int B_size = B_num_rows * B_num_cols;
    int C_size = A_num_rows * B_num_cols;

    const int64_t *hA_rows = a_cooRowInd_ptr;
    const int64_t *hA_columns = a_cooColInd_ptr;
    const float *hA_values = a_cooValues_ptr;
    const float *hB = b_ptr;
    float alpha = 1.0f;
    float beta = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int64_t *dA_rows, *dA_columns;
    float *dA_values, *dB, *dC;
    OF_CUDA_CHECK(cudaMalloc((void **)&dA_rows, A_nnz * sizeof(int64_t)));
    OF_CUDA_CHECK(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int64_t)));
    OF_CUDA_CHECK(cudaMalloc((void **)&dA_values, A_nnz * sizeof(float)));
    OF_CUDA_CHECK(cudaMalloc((void **)&dB, B_size * sizeof(float)));
    OF_CUDA_CHECK(cudaMalloc((void **)&dC, C_size * sizeof(float)));

    OF_CUDA_CHECK(cudaMemcpy(dA_rows, hA_rows, A_nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    OF_CUDA_CHECK(
        cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    OF_CUDA_CHECK(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice));
    OF_CUDA_CHECK(cudaMemcpy(dB, hB, B_size * sizeof(float), cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in COO format
    CHECK_CUSPARSE(cusparseCreateCoo(&matA, A_num_rows, A_num_cols, A_nnz, dA_rows, dA_columns,
                                     dA_values, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F))
    // Create dense matrix B
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW))
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, out_ptr, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
        matB, &beta, matC, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, &bufferSize))
    OF_CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    // execute SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, dBuffer))

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device memory deallocation
    OF_CUDA_CHECK(cudaFree(dBuffer));
    OF_CUDA_CHECK(cudaFree(dA_rows));
    OF_CUDA_CHECK(cudaFree(dA_columns));
    OF_CUDA_CHECK(cudaFree(dA_values));
    OF_CUDA_CHECK(cudaFree(dB));
    OF_CUDA_CHECK(cudaFree(dC));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SPMM_COO_KERNEL(device, dtype)            \
  REGISTER_USER_KERNEL("spmm_coo")                         \
      .SetCreateFn<SpmmCOOGpuFloatKernel>()                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_SPMM_COO_KERNEL(DeviceType::kGPU, float)
}  // namespace
}  // namespace oneflow
