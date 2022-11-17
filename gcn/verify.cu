#include <cusparse.h>
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE
#include <iostream>

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

  
  
  int   A_num_rows   = 5; // [5,4]  A^T = [4, 5]
  int   A_num_cols   = 4;
  bool  transpose_a  = true;
  int   A_nnz        = 9;
  int   B_num_rows   = 5; // [5, 3]
  int   B_num_cols   = 3;

/*   if(transpose_a) {
    std::swap(A_num_rows, A_num_cols);
  } */ 
  int   ldb          = B_num_rows; // 5
  int   ldc          = transpose_a? A_num_rows: A_num_cols; // 4
  int   B_size       = ldb * B_num_cols;  // 5 * 3 
  int   C_size       = ldc * B_num_cols; //  4 * 3  [4, 5][5, 3]
  //int   hA_rows[]    = { 0, 0, 0, 1, 2, 2, 2, 3, 3 };
  int   hA_rows[]    = { 0, 3, 4, 7, 9, 9 };
  
  int   hA_columns[] = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
  float hA_values[]  = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         6.0f, 7.0f, 8.0f, 9.0f };
  float  hB[]        = { 1.0f,  2.0f,  3.0f,  4.0f,
                         5.0f,  6.0f,  7.0f,  8.0f,
                         9.0f, 10.0f, 11.0f, 12.0f, 0, 0, 0 };
  float  hC[]        = { 0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0 };
  float  hC_result[] = { 19.0f,  8.0f,  51.0f,  52.0f,
                         43.0f, 24.0f, 123.0f, 120.0f,
                         67.0f, 40.0f, 195.0f, 188.0f };
  float  alpha       = 1.0f;
  float  beta        = 0.0f;
  //--------------------------------------------------------------------------
  // Device memory management
  int   *dA_rows, *dA_columns;
  float *dA_values, *dB, *dC;
  CHECK_CUDA( cudaMalloc((void**) &dA_rows,    (A_num_rows+1)*sizeof(int)))
  CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
  CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  )
  CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

  CHECK_CUDA( cudaMemcpy(dA_rows, hA_rows, (A_num_rows+1) * sizeof(int),
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
  cusparseOperation_t opa = convertTransToCusparseOperation(transpose_a);
  cusparseOperation_t opb = convertTransToCusparseOperation(transpose_b);


  int64_t ma = A_num_rows, ka = A_num_cols;
  if (transpose_a != 'n') std::swap(ma, ka);

  cusparseSpMatDescr_t matA;
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(
    &matA,                     /* output */
    ma, ka, nnz,                /* rows, cols, number of non zero elements */
    hA_rows,                    /* row offsets of the sparse matrix, size = rows +1 */
    hA_columns,                 /* column indices of the sparse matrix, size = nnz */
    hA_values,                    /* values of the sparse matrix, size = nnz */
    CUSPARSE_INDEX_32I,         /* data type of row offsets index */
    CUSPARSE_INDEX_32I,         /* data type of col indices */
    CUSPARSE_INDEX_BASE_ZERO,   /* base index of row offset and col indes */
    cusparse_value_type         /* data type of values */
  ));

  int64_t kb = B_num_rows, nb = B_num_cols;
  if (transpose_b != 'n') std::swap(kb, nb);

  cusparseDnMatDescr_t matB;
  TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
    &matB,               /* output */
    kb, nb, ldb,          /* rows, cols, leading dimension */
    b,                    /* values */
    cusparse_value_type,  /* data type of values */
    CUSPARSE_ORDER_COL    /* memory layout, ONLY column-major is supported now */
  ));

  cusparseDnMatDescr_t matC;
  TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
    &matC,               /* output */
    m, n, ldc,            /* rows, cols, leading dimension */
    c,                    /* values */ 
    cusparse_value_type,  /* data type of values */ 
    CUSPARSE_ORDER_COL    /* memory layout, ONLY column-major is supported now */
  ));


  auto handle = at::cuda::getCurrentCUDASparseHandle();

  // cusparseSpMM_bufferSize returns the bufferSize that can be used by cusparseSpMM
  size_t bufferSize;
  TORCH_CUDASPARSE_CHECK(cusparseSpMM_bufferSize(
    handle, opa, opb,
    &alpha,
    matA, matB,
    &beta,
    matC,
    cusparse_value_type,  /* data type in which the computation is executed */
    CUSPARSE_CSRMM_ALG1,  /* default computing algorithm for CSR sparse matrix format */
    &bufferSize           /* output */
  ));

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(bufferSize);

  TORCH_CUDASPARSE_CHECK(cusparseSpMM(
    handle, opa, opb,
    &alpha,
    matA, matB,
    &beta,
    matC,
    cusparse_value_type,  /* data type in which the computation is executed */
    CUSPARSE_CSRMM_ALG1,  /* default computing algorithm for CSR sparse matrix format */
    dataPtr.get()         /* external buffer */
  ));

  TORCH_CUDASPARSE_CHECK(cusparseDestroySpMat(matA));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(matB));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(matC));
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