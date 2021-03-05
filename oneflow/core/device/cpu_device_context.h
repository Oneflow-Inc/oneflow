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
#include "oneflow/core/vm/cpu_allocator.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

class CpuDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDeviceCtx);
  CpuDeviceCtx() = default;
  ~CpuDeviceCtx() = default;

  std::unique_ptr<DeviceCtx> Copy() const { return std::unique_ptr<DeviceCtx>(new CpuDeviceCtx()); }

  void SyncDevice() override {}
  void AddCallBack(std::function<void()> callback) const override { callback(); }

  vm::Allocator* mut_allocator() override { return Global<vm::CpuAllocator>::Get(); }

 private:
};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
