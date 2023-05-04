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
#include "oneflow/cambricon/ep/mlu_stream.h"

#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/common/mlu_guard.h"
#include "oneflow/cambricon/ep/mlu_device.h"
#include "oneflow/cambricon/ep/mlu_event.h"
#include "oneflow/core/hardware/node_device_descriptor_manager.h"
#include "oneflow/core/vm/bin_allocator.h"
#include "oneflow/core/vm/ep_backend_allocator.h"
#include "oneflow/core/vm/ep_backend_host_allocator.h"
#include "oneflow/core/vm/thread_safe_guard.h"

namespace oneflow {
namespace ep {

MluStream::MluStream(MluDevice* device) : device_index_(device->device_index()), device_(device) {
  MluCurrentDeviceGuard guard(device_index_);
  OF_MLU_CHECK(cnrtQueueCreate(&mlu_stream_));
  // handle is a pointer to cnnlContext struct that holds the Cambricon CNNL context.
  // see:https://www.cambricon.com/docs/sdk_1.10.0/cambricon_cnnl_1.15.2/developer_guide/cnnl_api/data/datatype.html#_CPPv412cnnlHandle_t
  OF_CNNL_CHECK(cnnlCreate(&cnnl_handle_));
  OF_CNNL_CHECK(cnnlSetQueue(cnnl_handle_, mlu_stream_));

  std::shared_ptr<ep::Device> ep_device(device, /*empty deleter*/ [](auto p) {});
  auto ep_backend_allocator =
      std::make_unique<vm::EpBackendAllocator>(ep_device, ep::AllocationOptions{});
  workspace_allocator_.reset(new vm::BinAllocator<vm::ThreadSafeLock>(
      ep::kMaxAlignmentRequirement, std::move(ep_backend_allocator)));

  auto ep_backend_host_allocator =
      std::make_unique<vm::EpBackendHostAllocator>(ep_device, ep::AllocationOptions{});
  host_workspace_allocator_.reset(new vm::BinAllocator<vm::ThreadSafeLock>(
      ep::kMaxAlignmentRequirement, std::move(ep_backend_host_allocator)));
}

MluStream::~MluStream() {
  MluCurrentDeviceGuard guard(device_index_);
  OF_MLU_CHECK(cnrtQueueSync(mlu_stream_));
  OF_CNNL_CHECK(cnnlDestroy(cnnl_handle_));
  OF_MLU_CHECK(cnrtQueueDestroy(mlu_stream_));
}

Maybe<void> MluStream::OnExecutionContextSetup() {
  OF_MLU_CHECK(cnrtSetDevice(device_index_));
  return Maybe<void>::Ok();
}

Maybe<void> MluStream::OnExecutionContextTeardown() { return Maybe<void>::Ok(); }

DeviceType MluStream::device_type() const { return DeviceType::kMLU; }

MluDevice* MluStream::device() const { return device_; }

Maybe<void> MluStream::Sync() {
  cnrtRet_t err = cnrtQueueSync(mlu_stream_);
  if (err == cnrtSuccess) {
    return Maybe<void>::Ok();
  } else {
    return Error::RuntimeError() << "MluStream::Sync error";
  }
}

void MluStream::RecordEvent(Event* event) {
  auto* mlu_event = static_cast<MluEvent*>(event);  // NOLINT
  OF_MLU_CHECK(cnrtPlaceNotifier(mlu_event->mlu_event(), mlu_stream_));
}

void MluStream::WaitEvent(Event* event) {
  auto* mlu_event = static_cast<MluEvent*>(event);  // NOLINT
  OF_MLU_CHECK(cnrtQueueWaitNotifier(mlu_event->mlu_event(), mlu_stream_, 0));
}

Maybe<void> MluStream::GetAsyncError() {
  cnrtRet_t err = cnrtGetLastError();
  if (err != cnrtSuccess) { return Error::RuntimeError() << "(" << cnrtGetErrorStr(err) << ")"; }
  return Maybe<void>::Ok();
}

cnrtQueue_t MluStream::mlu_stream() const { return mlu_stream_; }

cnnlHandle_t MluStream::cnnl_handle() const { return cnnl_handle_; }

}  // namespace ep
}  // namespace oneflow
