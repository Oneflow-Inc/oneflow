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
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/vm/stream_create_stream_policy.h"
#include "oneflow/core/framework/stream_on_independent_thread.h"

namespace oneflow {
namespace vm {

void Stream::__Init__(ThreadCtx* thread_ctx, Symbol<Device> device, StreamType stream_type,
                      const intrusive::shared_ptr<Dependence>& schedule_local_dep_object,
                      const std::vector<intrusive::shared_ptr<Dependence>>& transport_dependences) {
  set_thread_ctx(thread_ctx);
  device_ = device;
  stream_type_ = stream_type;
  stream_policy_ = CHECK_JUST(CreateStreamPolicy::Visit(stream_type, device));
  schedule_local_dep_object_ = schedule_local_dep_object;
  transport_dependences_ = transport_dependences;
  on_scheduler_thread_ = stream_policy_->OnSchedulerThread(stream_type);
}

int64_t Stream::device_id() const { return device_->device_id(); }

char* Stream::CheckSizeAndGetTmpSmallPinnedMemPtr(size_t size) {
  static constexpr int kSmallSize = 512;
  CHECK_LE(size, kSmallSize);
  if (!static_cast<bool>(small_pinned_mem_ptr_)) {
    auto* ep_device = stream_policy_->stream()->device();
    void* mem_ptr = nullptr;
    CHECK_JUST(ep_device->AllocPinned(ep::AllocationOptions{}, &mem_ptr, kSmallSize));
    std::function<void(char*)> Deleter = [ep_device](char* ptr) {
      ep_device->FreePinned(ep::AllocationOptions{}, ptr);
    };
    char* ptr = reinterpret_cast<char*>(mem_ptr);
    small_pinned_mem_ptr_ = decltype(small_pinned_mem_ptr_)(ptr, Deleter);
  }
  return small_pinned_mem_ptr_.get();
}

}  // namespace vm
}  // namespace oneflow
