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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <nccl.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include "oneflow/core/device/cuda_pseudo_half.h"

#if CUDA_VERSION >= 10020

#include <nvjpeg.h>

#endif

namespace oneflow {

const char* CublasGetErrorString(cublasStatus_t error);

const char* CurandGetErrorString(curandStatus_t error);

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

#define OF_CURAND_CHECK(condition)                                                                 \
  for (curandStatus_t _of_curand_check_status = (condition);                                       \
       _of_curand_check_status != CURAND_STATUS_SUCCESS;)                                          \
  LOG(FATAL) << "Check failed: " #condition " : " << CurandGetErrorString(_of_curand_check_status) \
             << " (" << _of_curand_check_status << ") "

#define OF_NCCL_CHECK(condition)                                                                \
  for (ncclResult_t _of_nccl_check_status = (condition); _of_nccl_check_status != ncclSuccess;) \
  LOG(FATAL) << "Check failed: " #condition " : " << ncclGetErrorString(_of_nccl_check_status)  \
             << " (" << _of_nccl_check_status << ") "

#if CUDA_VERSION >= 10020

#define OF_NVJPEG_CHECK(condition)                                                                 \
  for (nvjpegStatus_t _of_nvjpeg_check_status = (condition);                                       \
       _of_nvjpeg_check_status != NVJPEG_STATUS_SUCCESS;)                                          \
  LOG(FATAL) << "Check failed: " #condition " : " << NvjpegGetErrorString(_of_nvjpeg_check_status) \
             << " (" << _of_nvjpeg_check_status << ") "

#endif

template<typename T>
void CudaCheck(T error);

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

int32_t GetSMCudaMaxBlocksNum();
void InitGlobalCudaDeviceProp();
bool IsCuda9OnTuringDevice();

inline int32_t BlocksNum4ThreadsNum(const int32_t n) {
  CHECK_GT(n, 0);
  return std::min((n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock, kCudaMaxBlocksNum);
}

inline int32_t SMBlocksNum4ThreadsNum(const int32_t n) {
  CHECK_GT(n, 0);
  return std::min((n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock,
                  GetSMCudaMaxBlocksNum());
}

#define RUN_CUDA_KERNEL(func, device_ctx_ptr, thread_num, ...)           \
  func<<<SMBlocksNum4ThreadsNum(thread_num), kCudaThreadsNumPerBlock, 0, \
         (device_ctx_ptr)->cuda_stream()>>>(__VA_ARGS__)

size_t GetAvailableGpuMemSize(int dev_id);

#define CUDA_WORK_TYPE_SEQ       \
  OF_PP_MAKE_TUPLE_SEQ(kCompute) \
  OF_PP_MAKE_TUPLE_SEQ(kCopyH2D) \
  OF_PP_MAKE_TUPLE_SEQ(kCopyD2H) \
  OF_PP_MAKE_TUPLE_SEQ(kNccl)    \
  OF_PP_MAKE_TUPLE_SEQ(kMix)     \
  OF_PP_MAKE_TUPLE_SEQ(kDecodeH2D)

enum class CudaWorkType {
#define DECLARE_CUDA_WORK_TYPE(type) type,
  OF_PP_FOR_EACH_TUPLE(DECLARE_CUDA_WORK_TYPE, CUDA_WORK_TYPE_SEQ)
};

inline size_t GetCudaWorkTypeSize() { return OF_PP_SEQ_SIZE(CUDA_WORK_TYPE_SEQ); }

void NumaAwareCudaMallocHost(int32_t dev, void** ptr, size_t size);

template<typename T>
void NumaAwareCudaMallocHost(int32_t dev, T** ptr, size_t size) {
  NumaAwareCudaMallocHost(dev, reinterpret_cast<void**>(ptr), size);
}

// Set the CPU affinity to the closest processor(s) of a particular GPU.
void CudaDeviceSetCpuAffinity(int32_t dev);

#define CUDA_DATA_TYPE_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(float, CUDA_R_32F)  \
  OF_PP_MAKE_TUPLE_SEQ(double, CUDA_R_64F) \
  OF_PP_MAKE_TUPLE_SEQ(float16, CUDA_R_16F)

cudaDataType_t GetCudaDataType(DataType);

template<typename T>
struct CudaDataType;

#define SPECIALIZE_CUDA_DATA_TYPE(type_cpp, type_cuda) \
  template<>                                           \
  struct CudaDataType<type_cpp> : std::integral_constant<cudaDataType_t, type_cuda> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_CUDA_DATA_TYPE, CUDA_DATA_TYPE_SEQ);
#undef SPECIALIZE_CUDA_DATA_TYPE

class CudaCurrentDeviceGuard final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaCurrentDeviceGuard)
  explicit CudaCurrentDeviceGuard(int32_t dev_id);
  CudaCurrentDeviceGuard();
  ~CudaCurrentDeviceGuard();

 private:
  int32_t saved_dev_id_ = -1;
};

int GetCudaSmVersion();

int GetCudaPtxVersion();

}  // namespace oneflow

#else

namespace oneflow {

enum class CudaWorkType {};

inline size_t GetCudaWorkTypeSize() { return 0; }

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
