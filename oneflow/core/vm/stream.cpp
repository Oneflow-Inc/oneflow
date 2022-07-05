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
#include "oneflow/core/vm/stream_get_stream_type.h"
#include "oneflow/core/framework/stream_on_independent_thread.h"

namespace oneflow {
namespace vm {

void Stream::__Init__(
    ThreadCtx* thread_ctx, Symbol<Device> device, StreamRole stream_role,
    const intrusive::shared_ptr<MirroredObject>& schedule_local_dep_object,
    const Optional<intrusive::shared_ptr<MirroredObject>>& transport_local_dep_object) {
  set_thread_ctx(thread_ctx);
  device_ = device;
  stream_role_ = stream_role;
  stream_type_ = CHECK_JUST(GetStreamType::Visit(stream_role, device->enum_type()));
  stream_type_->InitDeviceCtx(mut_device_ctx(), this);
  schedule_local_dep_object_ = schedule_local_dep_object;
  transport_local_dep_object_ = transport_local_dep_object;
  on_scheduler_thread_ = stream_type_->OnSchedulerThread(stream_role);
}

int64_t Stream::device_id() const { return device_->device_id(); }

const StreamType& Stream::stream_type() const { return *stream_type_; }

}  // namespace vm
}  // namespace oneflow
