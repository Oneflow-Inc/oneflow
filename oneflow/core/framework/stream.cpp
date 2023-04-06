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
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/stream_is_comm_net_stream.h"
#include "oneflow/core/thread/thread_global_id.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/static_global.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/stream_mgr.h"
#include "oneflow/core/vm/stream_get_allocator_stream_type.h"

namespace oneflow {

Stream::Stream(Symbol<Device> device, StreamType stream_type, size_t thread_uid)
    : device_(device), stream_type_(stream_type), thread_uid_(thread_uid), unique_stream_id_(-1) {}

Maybe<void> Stream::Init(size_t unique_stream_id) {
  unique_stream_id_ = unique_stream_id;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<Symbol<Stream>> Stream::RawNew(Symbol<Device> device, StreamType stream_type,
                                                size_t thread_uid) {
  std::shared_ptr<Stream> stream(new Stream(device, stream_type, thread_uid));
  return JUST(SingletonMaybe<StreamMgr>())
      ->AddStreamSymbol(*stream, [&](size_t unique_stream_id) -> Maybe<Symbol<Stream>> {
        JUST(stream->Init(unique_stream_id));
        return SymbolOf(*stream);
      });
}

/*static*/ Maybe<Symbol<Stream>> Stream::New(Symbol<Device> device, StreamType stream_type,
                                             size_t thread_uid) {
  constexpr auto* Make = DECORATE(&Stream::RawNew, ThreadLocalCopiable);
  return Make(device, stream_type, thread_uid);
}

namespace {

Maybe<Symbol<Stream>> RawGetDefaultStreamByDevice(Symbol<Device> device) {
  return Stream::New(device, StreamType::kCompute);
}

Maybe<Symbol<Stream>> RawGetDefaultStreamByPlacement(Symbol<ParallelDesc> parallel_desc) {
  return RawGetDefaultStreamByDevice(JUST(GetTensorDevice(parallel_desc)));
}

Maybe<Symbol<Stream>> RawGetAllocatorStream(Symbol<Stream> stream) {
  StreamType allocator_stream_type = JUST(GetAllocatorStreamType::Visit(stream->stream_type()));
  if (allocator_stream_type == stream->stream_type()) { return stream; }
  return Stream::New(stream->device(), allocator_stream_type, stream->thread_uid());
}

}  // namespace

int64_t Stream::kDefaultStreamThreadUid = 0;

decltype(GetDefaultStreamByDevice) GetDefaultStreamByDevice =
    DECORATE(&RawGetDefaultStreamByDevice, ThreadLocal);

decltype(GetDefaultStreamByPlacement) GetDefaultStreamByPlacement =
    DECORATE(&RawGetDefaultStreamByPlacement, ThreadLocal);

decltype(GetAllocatorStream) GetAllocatorStream = DECORATE(&RawGetAllocatorStream, ThreadLocal);

}  // namespace oneflow
