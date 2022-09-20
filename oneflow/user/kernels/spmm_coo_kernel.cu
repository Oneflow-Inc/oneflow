#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cuda_util.h"

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


template<typename T>
class CudaSpmmCooKernel final : public user_op::OpKernel {
  public:
  CudaSpmmCooKernel() = default;
  ~CudaSpmmCooKernel() = default;

  private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor *a_coo_row = ctx->Tensor4ArgNameAndIndex("a_coo_row", 0);
    const user_op::Tensor *a_coo_col = ctx->Tensor4ArgNameAndIndex("a_coo_col", 0);
    const user_op::Tensor *a_coo_val = ctx->Tensor4ArgNameAndIndex("a_coo_val", 0);
    const user_op::Tensor *b = ctx->Tensor4ArgNameAndIndex("b", 0);

    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int32_t *dA_rows = a_coo_row->dptr<int32_t>(); 
    const int32_t *dA_columns = a_coo_col->dptr<int32_t>();
    const float *dA_values = a_coo_val->dptr<float>();
    const float *dB = b->dptr<float>();

    const auto A_num_rows = ctx->Attr<int64_t>("a_num_rows");
    const auto A_num_cols = ctx->Attr<int64_t>("a_num_cols");
    const auto A_nnz = a_coo_row->shape_view().elem_cnt();
    const auto B_num_rows = b->shape_view().At(0);
    const auto B_num_cols = b->shape_view().At(1); 

    int ldb = B_num_cols;
    int ldc = B_num_cols;
    int B_size = B_num_rows * B_num_cols;
    int C_size = A_num_rows * B_num_cols;
    float *dC = out_tensor->mut_dptr<float>();

    float alpha = 1.0f;
    float beta = 0.0f;

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    CHECK_CUSPARSE(cusparseCreate(&handle)) // inside stream
    // Create sparse matrix A in COO format
    CHECK_CUSPARSE(cusparseCreateCoo(&matA, A_num_rows, A_num_cols, A_nnz, 
                                   const_cast<int32_t*>(dA_rows), const_cast<int32_t*>(dA_columns), const_cast<float*>(dA_values),
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                      CUDA_R_32F))
    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, const_cast<float*>(dB),
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW))
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                       CUDA_R_32F,CUSPARSE_ORDER_ROW))

    // execute SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, nullptr))

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))

  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_SPMM_COO_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("spmm_coo")                                                             \
      .SetCreateFn<CudaSpmmCooKernel<dtype>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                       \
                        && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_SPMM_COO_KERNEL(float);
REGISTER_CUDA_SPMM_COO_KERNEL(double);
REGISTER_CUDA_SPMM_COO_KERNEL(int8_t);
REGISTER_CUDA_SPMM_COO_KERNEL(uint8_t);
REGISTER_CUDA_SPMM_COO_KERNEL(int32_t);
REGISTER_CUDA_SPMM_COO_KERNEL(int64_t);

}  // namespace oneflow