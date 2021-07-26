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
#include "oneflow/core/device/rocm_util.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/device/node_device_descriptor_manager.h"
#include "oneflow/core/device/rocm_device_descriptor.h"

namespace oneflow {

#ifdef WITH_HIP

void InitGlobalRocmDeviceProp() {
  CHECK(Global<hipDeviceProp_t>::Get() == nullptr) << "initialized Global<hipDeviceProp_t> twice";
  Global<hipDeviceProp_t>::New();
  hipGetDeviceProperties(Global<hipDeviceProp_t>::Get(), 0);
}

int32_t GetSMRocmMaxBlocksNum() {
  const auto& global_device_prop = *Global<hipDeviceProp_t>::Get();
  int32_t n =
      global_device_prop.multiProcessorCount * global_device_prop.maxThreadsPerMultiProcessor;
  return (n + kRocmThreadsNumPerBlock - 1) / kRocmThreadsNumPerBlock;
}

template<>
void RocmCheck(hipError_t error) {
  CHECK_EQ(error, hipSuccess) << hipGetErrorString(error);
}

size_t GetAvailableGpuMemSize(int dev_id) {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, dev_id);
  return prop.totalGlobalMem;
}

namespace {

std::function<void(void**, size_t)> GetRocmMallocHostFn(int32_t dev) {
  auto default_fn = [](void** ptr, size_t size) { hipHostMalloc(ptr, size); };
  auto manager = Global<device::NodeDeviceDescriptorManager>::Get();
  if (manager == nullptr) { return default_fn; }
  auto node_desc = manager->GetLocalNodeDeviceDescriptor();
  auto rocm_device = std::dynamic_pointer_cast<const device::RocmDeviceDescriptor>(
      node_desc->GetDevice(device::kRocmDeviceDescriptorClassName, dev));
  if (!rocm_device) { return default_fn; }
  auto saved_affinity = node_desc->Topology()->GetMemoryAffinity();
  if (!saved_affinity) { return default_fn; }
  auto device_affinity =
      node_desc->Topology()->GetMemoryAffinityByPCIBusID(rocm_device->PCIBusID());
  if (!device_affinity) { return default_fn; }
  return [device_affinity, saved_affinity, node_desc, default_fn](void** ptr, size_t size) {
    node_desc->Topology()->SetMemoryAffinity(device_affinity);
    default_fn(ptr, size);
    node_desc->Topology()->SetMemoryAffinity(saved_affinity);
  };
}

}  // namespace

void NumaAwareRocmMallocHost(int32_t dev, void** ptr, size_t size) {
  auto fn = GetRocmMallocHostFn(dev);
  fn(ptr, size);
}

hipblasDatatype_t GetRocmDataType(DataType val) {
#define MAKE_ENTRY(type_cpp, type_rocm) \
  if (val == GetDataType<type_cpp>::value) { return type_rocm; }
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ROCM_DATA_TYPE_SEQ);
#undef MAKE_ENTRY
  UNIMPLEMENTED();
}

RocmCurrentDeviceGuard::RocmCurrentDeviceGuard(int32_t dev_id) {
  OF_ROCM_CHECK(hipGetDevice(&saved_dev_id_));
  OF_ROCM_CHECK(hipSetDevice(dev_id));
}

RocmCurrentDeviceGuard::RocmCurrentDeviceGuard() { OF_ROCM_CHECK(hipGetDevice(&saved_dev_id_)); }

RocmCurrentDeviceGuard::~RocmCurrentDeviceGuard() { OF_ROCM_CHECK(hipSetDevice(saved_dev_id_)); }

#endif  // WITH_HIP

}  // namespace oneflow
