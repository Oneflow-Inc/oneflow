// NOTE(jiyuan): Adapted from Caffe
#ifndef _DEVICE_DEVICE_ALTERNATE_H_
#define _DEVICE_DEVICE_ALTERNATE_H_

#include <cstdint>
#include <glog/logging.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#include "common/cudnn_utils.h"

//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

// CUDA: various checks for different function calls.
    /*
    const char ** error_string; \
    if (error != cudaSuccess) { \
    cuGetErrorString(error, error_string); \
    } \
    */
#define CUDA_DRIVER_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    CUresult error = condition; \
    CHECK_EQ(error, cudaSuccess) << " CUDA_DRIVER_CHECK ERROR:"; \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
    const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
// CURAND macro
#define CURAND_CHECK(condition) \
do {\
	curandStatus_t status = condition; \
	CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
	<< caffe::curandGetErrorString(status); \
} while (0)


namespace caffe {
// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// NOTE(jiyuan): Adapted from CUDA V6.5 sample code "helper_cuda.h"
int32_t _ConvertSMVer2Cores(int32_t major, int32_t minor);
}  // namespace caffe

#endif  // _DEVICE_DEVICE_ALTERNATE_H_
