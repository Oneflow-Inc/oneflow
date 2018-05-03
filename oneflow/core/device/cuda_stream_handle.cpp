#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA

const cudaStream_t* CudaStreamHandle::cuda_stream() {
  if (!cuda_stream_) {
    cuda_stream_.reset(new cudaStream_t);
    CudaCheck(cudaStreamCreate(cuda_stream_.get()));
  }
  return cuda_stream_.get();
}

const cublasHandle_t* CudaStreamHandle::cublas_pmh_handle() {
  if (!cublas_pmh_handle_) {
    cublas_pmh_handle_.reset(new cublasHandle_t);
    CudaCheck(cublasCreate(cublas_pmh_handle_.get()));
    CudaCheck(cublasSetStream(*cublas_pmh_handle_, *cuda_stream()));
  }
  return cublas_pmh_handle_.get();
}

const cublasHandle_t* CudaStreamHandle::cublas_pmd_handle() {
  if (!cublas_pmd_handle_) {
    cublas_pmd_handle_.reset(new cublasHandle_t);
    CudaCheck(cublasCreate(cublas_pmd_handle_.get()));
    CudaCheck(cublasSetStream(*cublas_pmd_handle_, *cuda_stream()));
    CudaCheck(cublasSetPointerMode(*cublas_pmd_handle_, CUBLAS_POINTER_MODE_DEVICE));
  }
  return cublas_pmd_handle_.get();
}

const cudnnHandle_t* CudaStreamHandle::cudnn_handle() {
  if (!cudnn_handle_) {
    cudnn_handle_.reset(new cudnnHandle_t);
    CudaCheck(cudnnCreate(cudnn_handle_.get()));
    CudaCheck(cudnnSetStream(*cudnn_handle_, *cuda_stream()));
  }
  return cudnn_handle_.get();
}

const Eigen::GpuDevice* CudaStreamHandle::eigen_gpu_device() {
  if (!eigen_gpu_device_) {
    eigen_cuda_stream_.reset(new Eigen::CudaStreamDevice(cuda_stream()));
    eigen_gpu_device_.reset(new Eigen::GpuDevice(eigen_cuda_stream_.get()));
  }
  return eigen_gpu_device_.get();
}

CudaStreamHandle::~CudaStreamHandle() {
  if (cudnn_handle_) { CudaCheck(cudnnDestroy(*cudnn_handle_)); }
  if (cublas_pmh_handle_) { CudaCheck(cublasDestroy(*cublas_pmh_handle_)); }
  if (cublas_pmd_handle_) { CudaCheck(cublasDestroy(*cublas_pmd_handle_)); }
  if (cuda_stream_) { CudaCheck(cudaStreamDestroy(*cuda_stream_)); }
  eigen_gpu_device_.reset();
  eigen_cuda_stream_.reset();
}

#endif  // WITH_CUDA

}  // namespace oneflow
