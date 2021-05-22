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
#include "oneflow/core/device/cuda_device_descriptor.h"
#include "oneflow/core/device/cuda_util.h"

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <json.hpp>

namespace oneflow {

namespace device {

namespace {

constexpr char kJsonKeyOrdinal[] = "ordinal";
constexpr char kJsonKeyName[] = "name";
constexpr char kJsonKeyTotalGlobalMemory[] = "total_global_memory_bytes";
constexpr char kJsonKeyClockRate[] = "clock_rate_khz";
constexpr char kJsonKeyComputeCapabilityMajor[] = "compute_capability_major";
constexpr char kJsonKeyComputeCapabilityMinor[] = "compute_capability_minor";
constexpr char kJsonKeyMemoryClockRate[] = "memory_clock_rate_khz";
constexpr char kJsonKeyMemoryBusWidth[] = "memory_bus_width_bit";

}  // namespace

struct CudaDeviceDescriptor::Impl {
  int32_t ordinal{};
  std::string name;
  size_t total_global_memory_bytes{};
  int32_t clock_rate_khz{};
  int32_t compute_capability_major{};
  int32_t compute_capability_minor{};
  int32_t memory_clock_rate_khz{};
  int32_t memory_bus_width_bit{};
};

CudaDeviceDescriptor::CudaDeviceDescriptor() { impl_.reset(new Impl()); }

CudaDeviceDescriptor::~CudaDeviceDescriptor() = default;

int32_t CudaDeviceDescriptor::Ordinal() const { return impl_->ordinal; }

const std::string& CudaDeviceDescriptor::Name() const { return impl_->name; }

size_t CudaDeviceDescriptor::GlobalMemorySizeBytes() const {
  return impl_->total_global_memory_bytes;
}

int32_t CudaDeviceDescriptor::ClockRateKHz() const { return impl_->clock_rate_khz; }

int32_t CudaDeviceDescriptor::ComputeCapabilityMajor() const {
  return impl_->compute_capability_major;
}

int32_t CudaDeviceDescriptor::ComputeCapabilityMinor() const {
  return impl_->compute_capability_minor;
}

int32_t CudaDeviceDescriptor::MemoryClockRateKHz() const { return impl_->memory_clock_rate_khz; }

int32_t CudaDeviceDescriptor::MemoryBusWidthBit() const { return impl_->memory_bus_width_bit; }

std::shared_ptr<const CudaDeviceDescriptor> CudaDeviceDescriptor::Query(int32_t ordinal) {
  cudaDeviceProp prop{};
  OF_CUDA_CHECK(cudaGetDeviceProperties(&prop, ordinal));
  auto* desc = new CudaDeviceDescriptor();
  desc->impl_->ordinal = ordinal;
  desc->impl_->name = prop.name;
  desc->impl_->total_global_memory_bytes = prop.totalGlobalMem;
  desc->impl_->clock_rate_khz = prop.clockRate;
  desc->impl_->compute_capability_major = prop.major;
  desc->impl_->compute_capability_minor = prop.minor;
  desc->impl_->memory_clock_rate_khz = prop.memoryClockRate;
  desc->impl_->memory_bus_width_bit = prop.memoryBusWidth;
  return std::shared_ptr<const CudaDeviceDescriptor>(desc);
}

void CudaDeviceDescriptor::Serialize(std::string* serialized) const {
  nlohmann::json json_object;
  json_object[kJsonKeyOrdinal] = impl_->ordinal;
  json_object[kJsonKeyName] = impl_->name;
  json_object[kJsonKeyTotalGlobalMemory] = impl_->total_global_memory_bytes;
  json_object[kJsonKeyClockRate] = impl_->clock_rate_khz;
  json_object[kJsonKeyComputeCapabilityMajor] = impl_->compute_capability_major;
  json_object[kJsonKeyComputeCapabilityMinor] = impl_->compute_capability_minor;
  json_object[kJsonKeyMemoryClockRate] = impl_->memory_clock_rate_khz;
  json_object[kJsonKeyMemoryBusWidth] = impl_->memory_bus_width_bit;
  *serialized = json_object.dump(2);
}

std::shared_ptr<const CudaDeviceDescriptor> CudaDeviceDescriptor::Deserialize(
    const std::string& serialized) {
  auto json_object = nlohmann::json::parse(serialized);
  auto* desc = new CudaDeviceDescriptor();
  desc->impl_->ordinal = json_object[kJsonKeyOrdinal];
  desc->impl_->name = json_object[kJsonKeyName];
  desc->impl_->total_global_memory_bytes = json_object[kJsonKeyTotalGlobalMemory];
  desc->impl_->clock_rate_khz = json_object[kJsonKeyClockRate];
  desc->impl_->compute_capability_major = json_object[kJsonKeyComputeCapabilityMajor];
  desc->impl_->compute_capability_minor = json_object[kJsonKeyComputeCapabilityMinor];
  desc->impl_->memory_clock_rate_khz = json_object[kJsonKeyMemoryClockRate];
  desc->impl_->memory_bus_width_bit = json_object[kJsonKeyMemoryBusWidth];
  return std::shared_ptr<const CudaDeviceDescriptor>(desc);
}

}  // namespace device

}  // namespace oneflow

#endif  // WITH_CUDA
