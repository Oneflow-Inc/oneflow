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
#include "oneflow/core/device/hip_util.hip.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/device/node_device_descriptor_manager.h"
#include "oneflow/core/device/hip_device_descriptor.hip.h"

namespace oneflow {

#ifdef WITH_HIP

const char* HipblasGetErrorString(hipblasStatus_t error) {
  switch (error) {
    case HIPBLAS_STATUS_SUCCESS: return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED: return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED: return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE: return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_ARCH_MISMATCH: return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_MAPPING_ERROR: return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR: return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED: return "HIPBLAS_STATUS_NOT_SUPPORTED";
  }
  return "Unknown hipblas status";
}

const char* HiprandGetErrorString(hiprandStatus_t error) {
  switch (error) {
    case HIPRAND_STATUS_SUCCESS: return "HIPRAND_STATUS_SUCCESS";
    case HIPRAND_STATUS_VERSION_MISMATCH: return "HIPRAND_STATUS_VERSION_MISMATCH";
    case HIPRAND_STATUS_NOT_INITIALIZED: return "HIPRAND_STATUS_NOT_INITIALIZED";
    case HIPRAND_STATUS_ALLOCATION_FAILED: return "HIPRAND_STATUS_ALLOCATION_FAILED";
    case HIPRAND_STATUS_TYPE_ERROR: return "HIPRAND_STATUS_TYPE_ERROR";
    case HIPRAND_STATUS_OUT_OF_RANGE: return "HIPRAND_STATUS_OUT_OF_RANGE";
    case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE: return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
    case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case HIPRAND_STATUS_LAUNCH_FAILURE: return "HIPRAND_STATUS_LAUNCH_FAILURE";
    case HIPRAND_STATUS_PREEXISTING_FAILURE: return "HIPRAND_STATUS_PREEXISTING_FAILURE";
    case HIPRAND_STATUS_INITIALIZATION_FAILED: return "HIPRAND_STATUS_INITIALIZATION_FAILED";
    case HIPRAND_STATUS_ARCH_MISMATCH: return "HIPRAND_STATUS_ARCH_MISMATCH";
    case HIPRAND_STATUS_INTERNAL_ERROR: return "HIPRAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown hiprand status";
}

void InitGlobalHipDeviceProp() {
  CHECK(Global<hipDeviceProp_t>::Get() == nullptr) << "initialized Global<hipDeviceProp_t> twice";
  Global<hipDeviceProp_t>::New();
  hipGetDeviceProperties(Global<hipDeviceProp_t>::Get(), 0);
}

int32_t GetSMHipMaxBlocksNum() {
  const auto& global_device_prop = *Global<hipDeviceProp_t>::Get();
  int32_t n =
      global_device_prop.multiProcessorCount * global_device_prop.maxThreadsPerMultiProcessor;
  return (n + kHipThreadsNumPerBlock - 1) / kHipThreadsNumPerBlock;
}

template<>
void HipCheck(hipError_t error) {
  CHECK_EQ(error, hipSuccess) << hipGetErrorString(error);
}

template<>
void HipCheck(miopenStatus_t error) {
  CHECK_EQ(error, miopenStatusSuccess) << miopenGetErrorString(error);
}

template<>
void HipCheck(hipblasStatus_t error) {
  CHECK_EQ(error, HIPBLAS_STATUS_SUCCESS) << HipblasGetErrorString(error);
}

template<>
void HipCheck(hiprandStatus_t error) {
  CHECK_EQ(error, HIPRAND_STATUS_SUCCESS) << HiprandGetErrorString(error);
}

size_t GetAvailableGpuMemSize(int dev_id) {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, dev_id);
  return prop.totalGlobalMem;
}

namespace {

std::function<void(void**, size_t)> GetHipMallocHostFn(int32_t dev) {
  auto default_fn = [](void** ptr, size_t size) { hipHostMalloc(ptr, size); };
  auto manager = Global<device::NodeDeviceDescriptorManager>::Get();
  if (manager == nullptr) { return default_fn; }
  auto node_desc = manager->GetLocalNodeDeviceDescriptor();
  auto hip_device = std::dynamic_pointer_cast<const device::HipDeviceDescriptor>(
      node_desc->GetDevice(device::kHipDeviceDescriptorClassName, dev));
  if (!hip_device) { return default_fn; }
  auto saved_affinity = node_desc->Topology()->GetMemoryAffinity();
  if (!saved_affinity) { return default_fn; }
  auto device_affinity =
      node_desc->Topology()->GetMemoryAffinityByPCIBusID(hip_device->PCIBusID());
  if (!device_affinity) { return default_fn; }
  return [device_affinity, saved_affinity, node_desc, default_fn](void** ptr, size_t size) {
    node_desc->Topology()->SetMemoryAffinity(device_affinity);
    default_fn(ptr, size);
    node_desc->Topology()->SetMemoryAffinity(saved_affinity);
  };
}

}  // namespace

void NumaAwareHipMallocHost(int32_t dev, void** ptr, size_t size) {
  auto fn = GetHipMallocHostFn(dev);
  fn(ptr, size);
}

hipblasDatatype_t GetHipDataType(DataType val) {
#define MAKE_ENTRY(type_cpp, type_hip) \
  if (val == GetDataType<type_cpp>::value) { return type_hip; }
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, HIP_DATA_TYPE_SEQ);
#undef MAKE_ENTRY
  UNIMPLEMENTED();
}

HipCurrentDeviceGuard::HipCurrentDeviceGuard(int32_t dev_id) {
  OF_HIP_CHECK(hipGetDevice(&saved_dev_id_));
  OF_HIP_CHECK(hipSetDevice(dev_id));
}

HipCurrentDeviceGuard::HipCurrentDeviceGuard() { OF_HIP_CHECK(hipGetDevice(&saved_dev_id_)); }

HipCurrentDeviceGuard::~HipCurrentDeviceGuard() { OF_HIP_CHECK(hipSetDevice(saved_dev_id_)); }

#endif  // WITH_HIP

}  // namespace oneflow
