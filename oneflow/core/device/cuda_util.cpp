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
#include <mutex>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/hardware/node_device_descriptor_manager.h"
#include "oneflow/core/hardware/cuda_device_descriptor.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

#ifdef WITH_CUDA

#include <cuda.h>

#endif  // WITH_CUDA

namespace oneflow {

#ifdef WITH_CUDA

const char* CublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
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
    case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

#if CUDA_VERSION >= 10020

const char* NvjpegGetErrorString(nvjpegStatus_t error) {
  switch (error) {
    case NVJPEG_STATUS_SUCCESS: return "NVJPEG_STATUS_SUCCESS";
    case NVJPEG_STATUS_NOT_INITIALIZED: return "NVJPEG_STATUS_NOT_INITIALIZED";
    case NVJPEG_STATUS_INVALID_PARAMETER: return "NVJPEG_STATUS_INVALID_PARAMETER";
    case NVJPEG_STATUS_BAD_JPEG: return "NVJPEG_STATUS_BAD_JPEG";
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";
    case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "NVJPEG_STATUS_ALLOCATOR_FAILURE";
    case NVJPEG_STATUS_EXECUTION_FAILED: return "NVJPEG_STATUS_EXECUTION_FAILED";
    case NVJPEG_STATUS_ARCH_MISMATCH: return "NVJPEG_STATUS_ARCH_MISMATCH";
    case NVJPEG_STATUS_INTERNAL_ERROR: return "NVJPEG_STATUS_INTERNAL_ERROR";
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
      return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
  }
  return "Unknown nvjpeg status";
}

#endif

size_t GetAvailableGpuMemSize(int dev_id) {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev_id);
  return prop.totalGlobalMem;
}

namespace {

std::function<cudaError_t(void**, size_t)> GetCudaMallocHostFn(int32_t dev) {
  auto default_fn = [](void** ptr, size_t size) { return cudaMallocHost(ptr, size); };
  auto manager = Global<hardware::NodeDeviceDescriptorManager>::Get();
  if (manager == nullptr) { return default_fn; }
  auto node_desc = manager->GetLocalNodeDeviceDescriptor();
  auto cuda_device = std::dynamic_pointer_cast<const hardware::CudaDeviceDescriptor>(
      node_desc->GetDevice(hardware::kCudaDeviceDescriptorClassName, dev));
  if (!cuda_device) { return default_fn; }
  auto saved_affinity = node_desc->Topology()->GetMemoryAffinity();
  if (!saved_affinity) { return default_fn; }
  auto device_affinity =
      node_desc->Topology()->GetMemoryAffinityByPCIBusID(cuda_device->PCIBusID());
  if (!device_affinity) { return default_fn; }
  return [device_affinity, saved_affinity, node_desc, default_fn](void** ptr, size_t size) {
    node_desc->Topology()->SetMemoryAffinity(device_affinity);
    cudaError_t err = default_fn(ptr, size);
    node_desc->Topology()->SetMemoryAffinity(saved_affinity);
    return err;
  };
}

}  // namespace

cudaError_t NumaAwareCudaMallocHost(int32_t dev, void** ptr, size_t size) {
  auto fn = GetCudaMallocHostFn(dev);
  return fn(ptr, size);
}

CudaCurrentDeviceGuard::CudaCurrentDeviceGuard(int32_t dev_id) {
  CHECK(!pthread_fork::IsForkedSubProcess()) << pthread_fork::kOfCudaNotSupportInForkedSubProcess;
  OF_CUDA_CHECK(cudaGetDevice(&saved_dev_id_));
  OF_CUDA_CHECK(cudaSetDevice(dev_id));
}

CudaCurrentDeviceGuard::CudaCurrentDeviceGuard() { OF_CUDA_CHECK(cudaGetDevice(&saved_dev_id_)); }

CudaCurrentDeviceGuard::~CudaCurrentDeviceGuard() { OF_CUDA_CHECK(cudaSetDevice(saved_dev_id_)); }

CublasMathModeGuard::CublasMathModeGuard(cublasHandle_t handle, cublasMath_t new_mode)
    : CublasMathModeGuard(handle) {
  SetMathMode(new_mode);
}

CublasMathModeGuard::CublasMathModeGuard(cublasHandle_t handle) : handle_(handle) {
  OF_CUBLAS_CHECK(cublasGetMathMode(handle_, &saved_mode_));
  new_mode_ = saved_mode_;
}

CublasMathModeGuard::~CublasMathModeGuard() {
  if (new_mode_ != saved_mode_) { OF_CUBLAS_CHECK(cublasSetMathMode(handle_, saved_mode_)); }
}

void CublasMathModeGuard::SetMathMode(cublasMath_t new_mode) {
  new_mode_ = new_mode;
  if (new_mode_ != saved_mode_) { OF_CUBLAS_CHECK(cublasSetMathMode(handle_, new_mode_)); }
}

void CudaSynchronize(int device_id) {
  CudaCurrentDeviceGuard dev_guard(device_id);
  OF_CUDA_CHECK(cudaDeviceSynchronize());
}

void SetCudaDeviceIndex(int device_id) { OF_CUDA_CHECK(cudaSetDevice(device_id)); }

int GetCudaDeviceIndex() { return GlobalProcessCtx::LocalRank(); }

int GetCudaDeviceCount() {
  /* static */ int cuda_device_count = 0;
  CudaCurrentDeviceGuard dev_guard(GetCudaDeviceIndex());
  OF_CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
  return cuda_device_count;
}

void InitCudaContextOnce(int device_id) {
  static int device_count = GetCudaDeviceCount();
  static std::vector<std::once_flag> init_flags = std::vector<std::once_flag>(device_count);
  if (LazyMode::is_enabled()) { return; }
  if (device_id == -1) { device_id = GetCudaDeviceIndex(); }
  std::call_once(init_flags[device_id], [&]() {
    OF_CUDA_CHECK(cudaSetDevice(device_id));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
  });
}

cudaError_t CudaDriverGetPrimaryCtxActive(int dev, int* active) {
#if CUDA_VERSION >= 11030
  CUdevice cu_device{};
  {
    CUresult (*fnCuDeviceGet)(CUdevice*, int) = nullptr;
    cudaError_t err =
        cudaGetDriverEntryPoint("cuDeviceGet", (void**)&fnCuDeviceGet, cudaEnableDefault);
    if (err != cudaSuccess) { return err; }
    CUresult result = fnCuDeviceGet(&cu_device, dev);
    if (result == CUDA_SUCCESS) {
      // do nothing
    } else if (result == CUresult::CUDA_ERROR_INVALID_DEVICE) {
      return cudaErrorInvalidDevice;
    } else {
      return cudaErrorUnknown;
    }
  }
  {
    CUresult (*fnCuDevicePrimaryCtxGetState)(CUdevice, unsigned int*, int*) = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint(
        "cuDevicePrimaryCtxGetState", (void**)&fnCuDevicePrimaryCtxGetState, cudaEnableDefault);
    if (err != cudaSuccess) { return err; }
    unsigned int flags{};
    CUresult result = fnCuDevicePrimaryCtxGetState(cu_device, &flags, active);
    if (result == CUDA_SUCCESS) {
      return cudaSuccess;
    } else {
      return cudaErrorUnknown;
    }
  }
#else
  return cudaErrorNotSupported;
#endif  // CUDA_VERSION < 11030
}

#endif  // WITH_CUDA

}  // namespace oneflow
