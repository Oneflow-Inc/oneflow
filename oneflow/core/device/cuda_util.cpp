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

#ifdef USE_CUDNN
namespace cudnn {

float DataType<float>::oneval = 1.0;
float DataType<float>::zeroval = 0.0;
const void* DataType<float>::one = static_cast<void*>(&DataType<float>::oneval);
const void* DataType<float>::zero =
    static_cast<void*>(&DataType<float>::zeroval);

double DataType<double>::oneval = 1.0;
double DataType<double>::zeroval = 0.0;
const void* DataType<double>::one =
    static_cast<void*>(&DataType<double>::oneval);
const void* DataType<double>::zero =
    static_cast<void*>(&DataType<double>::zeroval);

/*
signed char DataType<signed char>::oneval = 1.0;
signed char DataType<signed char>::zeroval = 0.0;
const void* DataType<signed char>::one =
    static_cast<void*>(&DataType<signed char>::oneval);
const void* DataType<signed char>::zero =
    static_cast<void*>(&DataType<signed char>::zeroval);

int DataType<int>::oneval = 1.0;
int DataType<int>::zeroval = 0.0;
const void* DataType<int>::one = static_cast<void*>(&DataType<int>::oneval);
const void* DataType<int>::zero = static_cast<void*>(&DataType<int>::zeroval);
*/

}  // namespace cudnn
#endif  // USE_CUDNN

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
