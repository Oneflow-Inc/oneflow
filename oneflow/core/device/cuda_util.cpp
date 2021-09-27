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
#ifdef WITH_CUDA

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/memory/memory_case_registry.h"
#include "oneflow/core/device/node_device_descriptor_manager.h"
#include "oneflow/core/device/cuda_device_descriptor.h"

namespace oneflow {

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

namespace {

std::function<void(void**, size_t)> GetCudaMallocHostFn(int32_t dev) {
  auto default_fn = [](void** ptr, size_t size) { cudaMallocHost(ptr, size); };
  auto manager = Global<device::NodeDeviceDescriptorManager>::Get();
  if (manager == nullptr) { return default_fn; }
  auto node_desc = manager->GetLocalNodeDeviceDescriptor();
  auto cuda_device = std::dynamic_pointer_cast<const device::CudaDeviceDescriptor>(
      node_desc->GetDevice(device::kCudaDeviceDescriptorClassName, dev));
  if (!cuda_device) { return default_fn; }
  auto saved_affinity = node_desc->Topology()->GetMemoryAffinity();
  if (!saved_affinity) { return default_fn; }
  auto device_affinity =
      node_desc->Topology()->GetMemoryAffinityByPCIBusID(cuda_device->PCIBusID());
  if (!device_affinity) { return default_fn; }
  return [device_affinity, saved_affinity, node_desc, default_fn](void** ptr, size_t size) {
    node_desc->Topology()->SetMemoryAffinity(device_affinity);
    default_fn(ptr, size);
    node_desc->Topology()->SetMemoryAffinity(saved_affinity);
  };
}

}  // namespace

void NumaAwareCudaMallocHost(int32_t dev, void** ptr, size_t size) {
  auto fn = GetCudaMallocHostFn(dev);
  fn(ptr, size);
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

namespace {

bool MatchCudaMemoryCase(const MemCase& mem_case) {
  if (mem_case.Attr<DeviceType>("device_type") == kGPU) { return true; }
  return false;
}

bool MatchCudaOrCudaPinnedHostMemoryCase(const MemCase& mem_case) {
  if (mem_case.Attr<DeviceType>("device_type")==kCPU && mem_case.HasAttr<MemCase>("cuda_pinned_mem")) {
    return true;
  } else if (mem_case.Attr<DeviceType>("device_type") == kGPU) {
    return true;
  }
  return false;
}

bool MatchCudaOrCudaPinnedHostMemCaseId(const MemCaseId& mem_case_id) {
  if (mem_case_id.device_type() == DeviceType::kCPU
      && mem_case_id.host_mem_page_locked_device_type() == DeviceType::kGPU) {
    return true;
  } else if (mem_case_id.device_type() == DeviceType::kGPU) {
    return true;
  }
  return false;
}

}  // namespace

REGISTER_MEM_CASE_ID_GENERATOR(DeviceType::kGPU)
    .SetMatcher(MatchCudaOrCudaPinnedHostMemoryCase)
    .SetGenerator([](const MemCase& mem_case) -> MemCaseId {
      DeviceType device_type = DeviceType::kInvalidDevice;
      MemCaseId::device_index_t device_index = 0;
      DeviceType page_locked_device_type = DeviceType::kInvalidDevice;
      bool used_by_network = false;
      // if(mem_case.Attr<DeviceType>(const std::string &attr_name))
      // if(mem_case.device_type == kCPU)
      if (mem_case.Attr<DeviceType>("device_type") == kCPU) {
        CHECK(mem_case.HasAttr<MemCase>("cuda_pinned_mem"));
        device_type = DeviceType::kCPU;
        page_locked_device_type = DeviceType::kGPU;
        device_index = mem_case.Attr<int64_t>("cuda_pinned_mem_device_id");
        if (mem_case.HasAttr<MemCase>("used_by_network")) {
          used_by_network = true;
        }
      } else {
        CHECK(mem_case.Attr<DeviceType>("device_type") == kGPU);
        device_type = DeviceType::kGPU;
        device_index = mem_case.Attr<int64_t>("device_id");
      }
      return MemCaseId{device_type, device_index, page_locked_device_type, used_by_network};
    });

REGISTER_MEM_CASE_ID_TO_PROTO(DeviceType::kGPU)
    .SetMatcher(MatchCudaOrCudaPinnedHostMemCaseId)
    .SetToProto([](const MemCaseId& mem_case_id, MemCase* mem_case) -> void {
      if (mem_case_id.device_type() == DeviceType::kCPU) {
        CHECK_EQ(mem_case_id.host_mem_page_locked_device_type(), DeviceType::kGPU);
        mem_case->SetAttr("device_type", kCPU);
        mem_case->SetAttr("cuda_pinned_mem_device_id", mem_case_id.device_index());
        if (mem_case_id.is_host_mem_registered_by_network()) {
          mem_case->SetAttr("used_by_network", true);
        }
      } else if (mem_case_id.device_type() == DeviceType::kGPU) {
        mem_case->SetAttr("device_id", mem_case_id.device_index());
      } else {
        UNIMPLEMENTED();
      }
    });

REGISTER_PAGE_LOCKED_MEM_CASE(DeviceType::kGPU)
    .SetMatcher(MatchCudaMemoryCase)
    .SetPageLocker([](const MemCase& mem_case, MemCase* page_locked_mem_case) -> void {
      CHECK(mem_case.Attr<DeviceType>("device_type") == kGPU);
      page_locked_mem_case->SetAttr("cuda_pinned_mem_device_id", mem_case.Attr<int64_t>("device_id"));
    });

REGISTER_PATCH_MEM_CASE(DeviceType::kGPU)
    .SetMatcher(MatchCudaOrCudaPinnedHostMemoryCase)
    .SetPatcher([](const MemCase& src_mem_case, MemCase* dst_mem_case) -> bool {
      if (src_mem_case.Attr<DeviceType>("device_type") == kCPU) {
        CHECK(src_mem_case.HasAttr<MemCase>("cuda_pinned_mem"));
        if (!(dst_mem_case->Attr<DeviceType>("device_type") == kCPU)) { return false; }
        if (dst_mem_case->HasAttr<MemCase>("page_lock_case")) {
          dst_mem_case->SetAttr("cuda_pinned_mem", src_mem_case.Attr<MemCase>("cuda_pinned_mem"));
        } else {
          return false;
        }
        if (src_mem_case.HasAttr<bool>("used_by_network") 
            && src_mem_case.Attr<bool>("used_by_network")) {
          dst_mem_case->SetAttr("used_by_network", true);
        }
      } else {
        CHECK(src_mem_case.Attr<DeviceType>("device_type") == kGPU);
        if (!(dst_mem_case->Attr<DeviceType>("device_type") == kGPU)) { return false; }
        if (src_mem_case.Attr<int64_t>("device_id")
            != dst_mem_case->Attr<int64_t>("device_id")) {
          return false;
        }
      }
      return true;
    });

}  // namespace oneflow

#endif  // WITH_CUDA
