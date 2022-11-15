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
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

template<typename T>
class CudaSpmmCsrKernel final : public user_op::OpKernel {
 public:
  CudaSpmmCsrKernel() = default;
  ~CudaSpmmCsrKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* a_csr_row = ctx->Tensor4ArgNameAndIndex("a_csr_row", 0);
    const user_op::Tensor* a_csr_col = ctx->Tensor4ArgNameAndIndex("a_csr_col", 0);
    const user_op::Tensor* a_csr_val = ctx->Tensor4ArgNameAndIndex("a_csr_val", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    bool transpose_a = ctx->Attr<bool>("transpose_a");
    bool transpose_b = ctx->Attr<bool>("transpose_b");

    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int32_t* dA_rows = a_csr_row->dptr<int32_t>();
    const int32_t* dA_columns = a_csr_col->dptr<int32_t>();
    const float* dA_values = a_csr_val->dptr<float>();
    const float* dB = b->dptr<float>();

    const auto A_num_rows = ctx->Attr<int64_t>("a_num_rows");
    const auto A_num_cols = ctx->Attr<int64_t>("a_num_cols");
    const auto A_nnz = a_csr_val->shape_view().elem_cnt();
    const auto B_num_rows = b->shape_view().At(0);
    const auto B_num_cols = b->shape_view().At(1);

    int ldb = B_num_cols;
    int ldc = B_num_cols;
    int B_size = B_num_rows * B_num_cols;
    int C_size = transpose_a? A_num_cols * B_num_cols: A_num_rows * B_num_cols;
    float* dC = out_tensor->mut_dptr<float>();

    float alpha = 1.0f;
    float beta = 0.0f;


    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*  dBuffer    = NULL;
    size_t bufferSize = 0;
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    // Create sparse matrix A in CSR format
    OF_CUSPARSE_CHECK(
        cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz, const_cast<int32_t*>(dA_rows),
                          const_cast<int32_t*>(dA_columns), const_cast<float*>(dA_values),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Create dense matrix B
    OF_CUSPARSE_CHECK(cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb,
                                          const_cast<float*>(dB), CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create dense matrix C
    OF_CUSPARSE_CHECK(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC, CUDA_R_32F,
                                          CUSPARSE_ORDER_ROW));
    // allocate an external buffer if needed
    OF_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      cuda_stream->cusparse_handle(),
      transpose_a ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matA, matB, &beta, matC, CUDA_R_32F,
      CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));

    OF_CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    OF_CUSPARSE_CHECK(cusparseSpMM(cuda_stream->cusparse_handle(),
    transpose_a ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, 
      &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    // destroy matrix/vector descriptors
    OF_CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    OF_CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
    OF_CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
    OF_CUDA_CHECK(cudaFree(dBuffer));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_SPMM_CSR_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("spmm_csr")                                     \
      .SetCreateFn<CudaSpmmCsrKernel<dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_SPMM_CSR_KERNEL(float);
REGISTER_CUDA_SPMM_CSR_KERNEL(double);

}  // namespace oneflow
