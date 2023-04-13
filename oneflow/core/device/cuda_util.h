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
#ifndef ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_

#include "oneflow/core/common/data_type.h"

#ifdef WITH_CUDA

#include <cublas_v2.h>
#if CUDA_VERSION >= 11000
#include <cusolverDn.h>
#endif
#include <cuda.h>
#if CUDA_VERSION >= 10010
#include <cublasLt.h>
#endif
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <nccl.h>
#include <cuda_fp16.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_half.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

#if CUDA_VERSION >= 10020

#include <nvjpeg.h>

#endif

namespace oneflow {

const char* CublasGetErrorString(cublasStatus_t error);

const char* CurandGetErrorString(curandStatus_t error);

#if CUDA_VERSION >= 11000
const char* CusovlerGetErrorString(cusolverStatus_t error);
#endif

#if CUDA_VERSION >= 10020

const char* NvjpegGetErrorString(nvjpegStatus_t error);

#endif

#define OF_CUDA_CHECK(condition)                                                               \
  for (cudaError_t _of_cuda_check_status = (condition); _of_cuda_check_status != cudaSuccess;) \
  LOG(FATAL) << "Check failed: " #condition " : " << cudaGetErrorString(_of_cuda_check_status) \
             << " (" << _of_cuda_check_status << ") "

#define OF_CUDNN_CHECK(condition)                                                                \
  for (cudnnStatus_t _of_cudnn_check_status = (condition);                                       \
       _of_cudnn_check_status != CUDNN_STATUS_SUCCESS;)                                          \
  LOG(FATAL) << "Check failed: " #condition " : " << cudnnGetErrorString(_of_cudnn_check_status) \
             << " (" << _of_cudnn_check_status << ") "

#define OF_CUBLAS_CHECK(condition)                                                                 \
  for (cublasStatus_t _of_cublas_check_status = (condition);                                       \
       _of_cublas_check_status != CUBLAS_STATUS_SUCCESS;)                                          \
  LOG(FATAL) << "Check failed: " #condition " : " << CublasGetErrorString(_of_cublas_check_status) \
             << " (" << _of_cublas_check_status << ") "

#if CUDA_VERSION >= 11000
#define OF_CUSOLVER_CHECK(condition)                                        \
  for (cusolverStatus_t _of_cusolver_check_status = (condition);            \
       _of_cusolver_check_status != CUSOLVER_STATUS_SUCCESS;)               \
    LOG(FATAL) << "Check failed: " #condition " : "                         \
               << CusovlerGetErrorString(_of_cusolver_check_status) << " (" \
               << _of_cusolver_check_status << ") ";
#endif

#define OF_CURAND_CHECK(condition)                                                                 \
  for (curandStatus_t _of_curand_check_status = (condition);                                       \
       _of_curand_check_status != CURAND_STATUS_SUCCESS;)                                          \
  LOG(FATAL) << "Check failed: " #condition " : " << CurandGetErrorString(_of_curand_check_status) \
             << " (" << _of_curand_check_status << ") "

#define OF_NCCL_CHECK(condition)                                                                \
  for (ncclResult_t _of_nccl_check_status = (condition); _of_nccl_check_status != ncclSuccess;) \
  LOG(FATAL) << "Check failed: " #condition " : " << ncclGetErrorString(_of_nccl_check_status)  \
             << " (" << _of_nccl_check_status << "). "                                          \
             << "To see more detail, please run OneFlow with system variable NCCL_DEBUG=INFO"

#define OF_NCCL_CHECK_OR_RETURN(condition)                                                         \
  for (ncclResult_t _of_nccl_check_status = (condition); _of_nccl_check_status != ncclSuccess;)    \
  return Error::CheckFailedError().AddStackFrame([](const char* function) {                        \
    thread_local static auto frame = SymbolOf(ErrorStackFrame(__FILE__, __LINE__, function));      \
    return frame;                                                                                  \
  }(__FUNCTION__))                                                                                 \
         << "Check failed: " #condition " : " << ncclGetErrorString(_of_nccl_check_status) << " (" \
         << _of_nccl_check_status << ") "

#if CUDA_VERSION >= 10020

#define OF_NVJPEG_CHECK(condition)                                                                 \
  for (nvjpegStatus_t _of_nvjpeg_check_status = (condition);                                       \
       _of_nvjpeg_check_status != NVJPEG_STATUS_SUCCESS;)                                          \
  LOG(FATAL) << "Check failed: " #condition " : " << NvjpegGetErrorString(_of_nvjpeg_check_status) \
             << " (" << _of_nvjpeg_check_status << ") "

#endif

// CUDA: grid stride looping
#define CUDA_1D_KERNEL_LOOP(i, n)                                                                 \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)

#define CUDA_1D_KERNEL_LOOP_T(type, i, n)                                                      \
  for (type i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)

const int32_t kCudaThreadsNumPerBlock = 512;
const int32_t kCudaMaxBlocksNum = 8192;
const int32_t kCudaWarpSize = 32;

// 48KB, max byte size of shared memroy per thread block
// TODO: limit of shared memory should be different for different arch
const int32_t kCudaMaxSharedMemoryByteSize = 48 << 10;

inline int64_t BlocksNum4ThreadsNum(const int64_t n) {
  CHECK_GT(n, 0);
  return std::min((n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock,
                  static_cast<int64_t>(kCudaMaxBlocksNum));
}

#define RUN_CUDA_KERNEL(func, stream, elem_cnt, ...) \
  stream->As<ep::CudaStream>()->LaunchKernel(func, elem_cnt, 1, __VA_ARGS__)

size_t GetAvailableGpuMemSize(int dev_id);

cudaError_t NumaAwareCudaMallocHost(int32_t dev, void** ptr, size_t size);

class CudaCurrentDeviceGuard final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaCurrentDeviceGuard);
  explicit CudaCurrentDeviceGuard(int32_t dev_id);
  CudaCurrentDeviceGuard();
  ~CudaCurrentDeviceGuard();

 private:
  int32_t saved_dev_id_ = -1;
};

class CublasMathModeGuard final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CublasMathModeGuard);
  CublasMathModeGuard(cublasHandle_t handle, cublasMath_t new_mode);
  explicit CublasMathModeGuard(cublasHandle_t handle);
  ~CublasMathModeGuard();

  void SetMathMode(cublasMath_t new_mode);

 private:
  cublasHandle_t handle_{};
  cublasMath_t saved_mode_{};
  cublasMath_t new_mode_{};
};

int GetCudaDeviceIndex();

int GetCudaDeviceCount();

Maybe<double> GetCUDAMemoryUsed();

cudaDeviceProp* GetDeviceProperties(int device_id);

void SetCudaDeviceIndex(int device_id);

void CudaSynchronize(int device_id);

void InitCudaContextOnce(int device_id);

cudaError_t CudaDriverGetPrimaryCtxActive(int dev, int* active);

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
