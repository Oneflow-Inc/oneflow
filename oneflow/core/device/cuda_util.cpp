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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/global.h"

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

void InitGlobalCudaDeviceProp() {
  CHECK(Global<cudaDeviceProp>::Get() == nullptr) << "initialized Global<cudaDeviceProp> twice";
  Global<cudaDeviceProp>::New();
  cudaGetDeviceProperties(Global<cudaDeviceProp>::Get(), 0);
  if (IsCuda9OnTuringDevice()) {
    LOG(WARNING)
        << "CUDA 9 running on Turing device has known issues, consider upgrading to CUDA 10";
  }
}

int32_t GetSMCudaMaxBlocksNum() {
  const auto& global_device_prop = *Global<cudaDeviceProp>::Get();
  int32_t n =
      global_device_prop.multiProcessorCount * global_device_prop.maxThreadsPerMultiProcessor;
  return (n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock;
}

bool IsCuda9OnTuringDevice() {
  const auto& global_device_prop = *Global<cudaDeviceProp>::Get();
  return CUDA_VERSION >= 9000 && CUDA_VERSION < 9020 && global_device_prop.major == 7
         && global_device_prop.minor == 5;
}

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

size_t GetAvailableGpuMemSize(int dev_id) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev_id);
  return prop.totalGlobalMem;
}

#ifdef OF_PLATFORM_POSIX

namespace {

void ParseCpuMask(const std::string& cpu_mask, cpu_set_t* cpu_set) {
  CPU_ZERO_S(sizeof(cpu_set_t), cpu_set);
  const char* const head = cpu_mask.c_str();
  const char* const tail = head + cpu_mask.size();
  const char* pos = head;
  std::vector<uint64_t> masks;
  while (pos < tail) {
    char* end_pos = nullptr;
    const uint64_t mask = std::strtoul(pos, &end_pos, 16);
    if (pos != head) {
      CHECK_EQ(end_pos - pos, 8);
    } else {
      CHECK_NE(end_pos, pos);
      CHECK_LE(end_pos - pos, 8);
    }
    if (end_pos < tail) { CHECK_EQ(*end_pos, ','); }
    masks.push_back(mask);
    pos = end_pos + 1;
  }
  int32_t cpu = 0;
  for (int64_t i = masks.size() - 1; i >= 0; i--) {
    for (uint64_t b = 0; b < 32; b++) {
      if ((masks.at(i) & (1UL << b)) != 0) { CPU_SET_S(cpu, sizeof(cpu_set_t), cpu_set); }
      cpu += 1;
    }
  }
}

std::string CudaDeviceGetCpuMask(int32_t dev_id) {
  std::vector<char> pci_bus_id_buf(sizeof("0000:00:00.0"));
  OF_CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id_buf.data(),
                                      static_cast<int>(pci_bus_id_buf.size()), dev_id));
  for (int32_t i = 0; i < pci_bus_id_buf.size(); ++i) {
    pci_bus_id_buf[i] = std::tolower(pci_bus_id_buf[i]);
  }
  const std::string pci_bus_id(pci_bus_id_buf.data(), pci_bus_id_buf.size() - 1);
  const std::string pci_bus_id_short = pci_bus_id.substr(0, sizeof("0000:00") - 1);
  const std::string local_cpus_file =
      "/sys/class/pci_bus/" + pci_bus_id_short + "/device/" + pci_bus_id + "/local_cpus";
  char* cpu_map_path = realpath(local_cpus_file.c_str(), nullptr);
  CHECK_NOTNULL(cpu_map_path);
  std::ifstream is(cpu_map_path);
  std::string cpu_mask;
  CHECK(std::getline(is, cpu_mask).good());
  is.close();
  free(cpu_map_path);
  return cpu_mask;
}

void CudaDeviceGetCpuAffinity(int32_t dev_id, cpu_set_t* cpu_set) {
  const std::string cpu_mask = CudaDeviceGetCpuMask(dev_id);
  ParseCpuMask(cpu_mask, cpu_set);
}

}  // namespace

#endif

void NumaAwareCudaMallocHost(int32_t dev, void** ptr, size_t size) {
#ifdef OF_PLATFORM_POSIX
  cpu_set_t new_cpu_set;
  CudaDeviceGetCpuAffinity(dev, &new_cpu_set);
  cpu_set_t saved_cpu_set;
  CHECK_EQ(sched_getaffinity(0, sizeof(cpu_set_t), &saved_cpu_set), 0);
  CHECK_EQ(sched_setaffinity(0, sizeof(cpu_set_t), &new_cpu_set), 0);
  OF_CUDA_CHECK(cudaMallocHost(ptr, size));
  CHECK_EQ(sched_setaffinity(0, sizeof(cpu_set_t), &saved_cpu_set), 0);
#else
  UNIMPLEMENTED();
#endif
}

void CudaDeviceSetCpuAffinity(int32_t dev) {
#ifdef OF_PLATFORM_POSIX
  cpu_set_t new_cpu_set;
  CudaDeviceGetCpuAffinity(dev, &new_cpu_set);
  CHECK_EQ(sched_setaffinity(0, sizeof(cpu_set_t), &new_cpu_set), 0);
#else
  UNIMPLEMENTED();
#endif
}

cudaDataType_t GetCudaDataType(DataType val) {
#define MAKE_ENTRY(type_cpp, type_cuda) \
  if (val == GetDataType<type_cpp>::value) { return type_cuda; }
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CUDA_DATA_TYPE_SEQ);
#undef MAKE_ENTRY
  UNIMPLEMENTED();
}

CudaCurrentDeviceGuard::CudaCurrentDeviceGuard(int32_t dev_id) {
  OF_CUDA_CHECK(cudaGetDevice(&saved_dev_id_));
  OF_CUDA_CHECK(cudaSetDevice(dev_id));
}

CudaCurrentDeviceGuard::CudaCurrentDeviceGuard() { OF_CUDA_CHECK(cudaGetDevice(&saved_dev_id_)); }

CudaCurrentDeviceGuard::~CudaCurrentDeviceGuard() { OF_CUDA_CHECK(cudaSetDevice(saved_dev_id_)); }

#endif  // WITH_CUDA

}  // namespace oneflow
