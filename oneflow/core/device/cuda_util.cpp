#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace {

const char* CublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char* CurandGetErrorString(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

}  // namespace

float CudnnDataType<float>::oneval = 1.0f;
float CudnnDataType<float>::zeroval = 0.0f;
const void* CudnnDataType<float>::one =
    static_cast<void*>(&CudnnDataType<float>::oneval);
const void* CudnnDataType<float>::zero =
    static_cast<void*>(&CudnnDataType<float>::zeroval);

double CudnnDataType<double>::oneval = 1.0;
double CudnnDataType<double>::zeroval = 0.0;
const void* CudnnDataType<double>::one =
    static_cast<void*>(&CudnnDataType<double>::oneval);
const void* CudnnDataType<double>::zero =
    static_cast<void*>(&CudnnDataType<double>::zeroval);

template<>
void CudaCheck(cudaError_t error) {
  CHECK_EQ(error, cudaSuccess) << cudaGetErrorString(error);
}

template<>
void CudaCheck(cudnnStatus_t error) {
  CHECK_EQ(error, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(error);
}

template<>
void CudaCheck(cublasStatus_t error) {
  CHECK_EQ(error, CUBLAS_STATUS_SUCCESS) << CublasGetErrorString(error);
}

template<>
void CudaCheck(curandStatus_t error) {
  CHECK_EQ(error, CURAND_STATUS_SUCCESS) << CurandGetErrorString(error);
}

}  // namespace oneflow
