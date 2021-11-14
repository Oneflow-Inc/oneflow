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

namespace oneflow {

class CpuDeviceCtx final : public DeviceCtx, public EventRecordProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDeviceCtx);
  CpuDeviceCtx()
  {
    onednn_engine_.reset(new dnnl::engine(dnnl::engine::kind::cpu, 0));
    onednn_stream_.reset(new dnnl::stream(*onednn_engine_));
  }
  ~CpuDeviceCtx() = default;

  std::unique_ptr<DeviceCtx> Copy() const { return std::unique_ptr<DeviceCtx>(new CpuDeviceCtx()); }

  void SyncDevice() override {}
  void AddCallBack(std::function<void()> callback) const override { callback(); }

  vm::Allocator* mut_allocator() override { return Global<vm::CpuAllocator>::Get(); }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  dnnl::engine* onednn_engine() const override { return onednn_engine_.get();};
  dnnl::stream* onednn_stream() const override { return onednn_stream_.get();};

  std::shared_ptr<EventRecord> MakeEventRecord() override {
    return std::make_shared<NaiveEventRecord>();
  }

 private:
  std::unique_ptr<dnnl::engine> onednn_engine_;
  std::unique_ptr<dnnl::stream> onednn_stream_;
};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
