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
#ifndef ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/event_record.h"
#include "oneflow/core/vm/cpu_allocator.h"
#include "oneflow/core/vm/cuda_host_allocator.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

class CpuDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDeviceCtx);
  CpuDeviceCtx() {
    device_ = Global<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCPU, 0);
    stream_ = device_->CreateStream();
  }
  ~CpuDeviceCtx() { device_->DestroyStream(stream_); }

  std::unique_ptr<DeviceCtx> Copy() const { return std::unique_ptr<DeviceCtx>(new CpuDeviceCtx()); }

  vm::Allocator* mut_allocator() override { return Global<vm::CpuAllocator>::Get(); }

  vm::Allocator* mut_pin_memory_allocator() { return Global<vm::CudaHostAllocator>::Get(); }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  ep::Stream* stream() override { return stream_; }

 private:
  std::shared_ptr<ep::Device> device_;
  ep::Stream* stream_;
};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
