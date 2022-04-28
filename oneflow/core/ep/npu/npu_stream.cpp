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
#include "oneflow/core/ep/npu/npu_stream.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/hardware/node_device_descriptor_manager.h"
#include "oneflow/core/ep/npu/npu_event.h"
#include "oneflow/core/ep/npu/npu_device.h"
#include <iostream>
#ifdef WITH_NPU

namespace oneflow {

namespace ep {

namespace {

constexpr size_t kDefaultWorkspaceSize = 4 * 1024 * 1024;  // 4M


}  // namespace


NpuStream::NpuStream(NpuDevice* device)
    : device_index_(device->device_index()), device_(device) {
  std::cout<<"NpuStream::NpuStream(NpuDevice* device)"<<std::endl;
  NpuCurrentDeviceGuard guard(device_index_);
  // npu_stream
  OF_NPU_CHECK(aclrtCreateStream(&npu_stream_));
  workspace_size_ = kDefaultWorkspaceSize;
  OF_NPU_CHECK(aclrtMalloc(&workspace_, workspace_size_,ACL_MEM_MALLOC_NORMAL_ONLY));
}

NpuStream::~NpuStream() {
  std::cout<<"NpuStream::~NpuStream()"<<std::endl;
  NpuCurrentDeviceGuard guard(device_index_);
  OF_NPU_CHECK(aclrtSynchronizeStream(npu_stream_));
  OF_NPU_CHECK(aclrtDestroyStream(npu_stream_));
  OF_NPU_CHECK(aclrtFree(workspace_));
}

Maybe<void> NpuStream::OnExecutionContextSetup() {
  OF_NPU_CHECK(aclrtSetDevice(device_index_));
  //SetAffinityByDevice(device_index_);
  std::cout<<"NpuStream::OnExecutionContextSetup SetAffinityByDevice Not Implement"<<std::endl;
  return Maybe<void>::Ok();
}

Maybe<void> NpuStream::OnExecutionContextTeardown() { return Maybe<void>::Ok(); }

DeviceType NpuStream::device_type() const { return DeviceType::kNPU; }

Device* NpuStream::device() const { return device_; }

Maybe<void> NpuStream::Sync() {
  std::cout<<"NpuStream::~Sync()"<<std::endl;
  aclError err = aclrtSynchronizeStream(npu_stream_);
  if (err == ACL_SUCCESS) {
    return Maybe<void>::Ok();
  } else {
    return Error::RuntimeError() <<" (" << err << ") ";
  }
}

void NpuStream::RecordEvent(Event* event) {
  std::cout<<"NpuStream::~RecordEvent()"<<std::endl;
  auto* npu_event = static_cast<NpuEvent*>(event);  // NOLINT
  OF_NPU_CHECK(aclrtRecordEvent(npu_event->npu_event(), npu_stream_));
}

aclrtStream NpuStream::npu_stream() const { return npu_stream_; }

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA
