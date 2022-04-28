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
#ifndef ONEFLOW_CORE_DEVICE_NPU_COPY_D2H_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_NPU_COPY_D2H_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/npu_event.h"
#include "oneflow/core/vm/npu_host_allocator.h"
#include "oneflow/core/ep/npu/npu_stream.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/ep/npu/npu_device.h"

namespace oneflow {
namespace vm {

#ifdef WITH_NPU

class NpuCopyD2HDeviceCtx : public DeviceCtx, public SingleThreadQueryNpuEventProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NpuCopyD2HDeviceCtx);
  NpuCopyD2HDeviceCtx() = delete;
  ~NpuCopyD2HDeviceCtx() override {
    if (stream_ != nullptr) {
      CHECK(device_);
      device_->DestroyStream(stream_);
    }
  }

  NpuCopyD2HDeviceCtx(int64_t device_id)
      : DeviceCtx(),
        SingleThreadQueryNpuEventProvider(device_id),
        stream_(nullptr),
        npu_allocator_(std::make_unique<NpuHostAllocator>(device_id)),
        device_id_(device_id) {}

  aclrtStream npu_stream() const override { return GetOrCreateNpuStream()->npu_stream(); }


  ep::Stream* stream() override { return GetOrCreateNpuStream(); }

  vm::Allocator* mut_allocator() override { return npu_allocator_.get(); }

  DeviceType device_type() const override { return DeviceType::kNPU; }

 private:
  ep::NpuStream* GetOrCreateNpuStream() const {
    if (unlikely(stream_ == nullptr)) {
      CHECK(!device_);
      device_ = std::dynamic_pointer_cast<ep::NpuDevice>(
          Global<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kNPU, device_id_));
      CHECK(device_);
      stream_ = dynamic_cast<ep::NpuStream*>(device_->CreateStream());
      CHECK(stream_ != nullptr);
    }
    return stream_;
  }

 protected:
  mutable std::shared_ptr<ep::NpuDevice> device_;
  mutable ep::NpuStream* stream_;
  std::unique_ptr<NpuHostAllocator> npu_allocator_;
  int64_t device_id_;
};

#endif  // WITH_NPU
}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NPU_COPY_D2H_DEVICE_CONTEXT_H_
