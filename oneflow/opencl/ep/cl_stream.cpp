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
#include "oneflow/opencl/ep/cl_stream.h"

#include "oneflow/opencl/common/cl_util.h"
#include "oneflow/opencl/common/cl_guard.h"
#include "oneflow/opencl/ep/cl_device.h"
#include "oneflow/opencl/ep/cl_event.h"
#include "oneflow/core/hardware/node_device_descriptor_manager.h"
#include "oneflow/core/vm/bin_allocator.h"
#include "oneflow/core/vm/ep_backend_allocator.h"
#include "oneflow/core/vm/ep_backend_host_allocator.h"
#include "oneflow/core/vm/thread_safe_guard.h"

namespace oneflow {
namespace ep {

clStream::clStream(clDevice* device) : device_index_(device->device_index()), device_(device) {
  clCurrentDeviceGuard guard(device_index_);
  OF_CL_CHECK(clQueueCreate(&cl_stream_));
}

clStream::~clStream() {
  clCurrentDeviceGuard guard(device_index_);
  OF_CL_CHECK(clQueueSynchronize(cl_stream_));
  OF_CL_CHECK(clQueueDestroy(cl_stream_));
}

Maybe<void> clStream::OnExecutionContextSetup() {
  OF_CL_CHECK(clSetDevice(device_index_));
  return Maybe<void>::Ok();
}

Maybe<void> clStream::OnExecutionContextTeardown() { return Maybe<void>::Ok(); }

DeviceType clStream::device_type() const { return DeviceType::kOpenCL; }

clDevice* clStream::device() const { return device_; }

Maybe<void> clStream::Sync() {
  cl_int err = clQueueSynchronize(cl_stream_);
  if (err == CL_SUCCESS) {
    return Maybe<void>::Ok();
  } else {
    return Error::RuntimeError() << "clStream::Sync error";
  }
}

void clStream::RecordEvent(Event* event) {
  auto* cl_event = static_cast<clEvent*>(event);  // NOLINT
  OF_CL_CHECK(clEventRecord(cl_event->cl_event(), cl_stream_));
}

void clStream::WaitEvent(Event* event) {
  auto* cl_event = static_cast<clEvent*>(event);  // NOLINT
  OF_CL_CHECK(clQueueWaitEvent(cl_event->cl_event(), cl_stream_, 0));
}

Maybe<void> clStream::GetAsyncError() {
  // TODO
  return Maybe<void>::Ok();
}

cl::CommandQueue* clStream::cl_stream() const { return cl_stream_; }

}  // namespace ep
}  // namespace oneflow
