#include <cusparse.h>
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}                                                                              \


#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
    }                                                                          \
  }


__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");

}


int main() {
  //  cuda_hello<<<1,1>>>(); 

  
  
  int   A_num_rows   = 4;
  int   A_num_cols   = 4;
  int   A_nnz        = 9;
  int   B_num_rows   = A_num_cols;
  int   B_num_cols   = 3;
  int   ldb          = B_num_rows;
  int   ldc          = A_num_rows;
  int   B_size       = ldb * B_num_cols;
  int   C_size       = ldc * B_num_cols;
  int   hA_rows[]    = { 0, 0, 0, 1, 2, 2, 2, 3, 3 };
  int   hA_columns[] = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
  float hA_values[]  = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         6.0f, 7.0f, 8.0f, 9.0f };
  float  hB[]        = { 1.0f,  2.0f,  3.0f,  4.0f,
                         5.0f,  6.0f,  7.0f,  8.0f,
                         9.0f, 10.0f, 11.0f, 12.0f };
  float  hC[]        = { 0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f };
  float  hC_result[] = { 19.0f,  8.0f,  51.0f,  52.0f,
                         43.0f, 24.0f, 123.0f, 120.0f,
                         67.0f, 40.0f, 195.0f, 188.0f };
  float  alpha       = 1.0f;
  float  beta        = 0.0f;
  //--------------------------------------------------------------------------
  // Device memory management
  int   *dA_rows, *dA_columns;
  float *dA_values, *dB, *dC;
  CHECK_CUDA( cudaMalloc((void**) &dA_rows,    A_nnz * sizeof(int))    )
  CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
  CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  )
  CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

  CHECK_CUDA( cudaMemcpy(dA_rows, hA_rows, A_nnz * sizeof(int),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                         cudaMemcpyHostToDevice) )
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t     handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  CHECK_CUSPARSE( cusparseCreate(&handle) )
  // Create sparse matrix A in COO format
  CHECK_CUSPARSE( cusparseCreateCoo(&matA, A_num_rows, A_num_cols, A_nnz,
                                    dA_rows, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  // Create dense matrix B
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB,
                                      CUDA_R_32F, CUSPARSE_ORDER_COL) )
  // Create dense matrix C
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                      CUDA_R_32F, CUSPARSE_ORDER_COL) )
  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                               handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                               CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
  CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

  // execute SpMM
  CHECK_CUSPARSE( cusparseSpMM(handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                               CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  //--------------------------------------------------------------------------
  // device result check
  CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),
                         cudaMemcpyDeviceToHost) )


    for(int i = 0; i < A_num_rows; i++) {
      for(int j = 0; j < B_num_cols; j++) {
        std::printf(" %f ", hC[i + j * ldc]);
      }
      std::printf("\n");

    }


    cudaFree(dA_rows);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dB);
    cudaFree(dC);
    printf("Hello World from CPU!\n");
    cudaDeviceSynchronize();
    return 0;
}