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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/static_global.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/stream_mgr.h"

namespace oneflow {

Stream::Stream(Symbol<Device> device, StreamRole stream_role)
    : device_(device), stream_role_(stream_role), unique_stream_id_(-1) {}

Maybe<void> Stream::Init(size_t unique_stream_id) {
  unique_stream_id_ = unique_stream_id;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<Symbol<Stream>> Stream::RawNew(Symbol<Device> device, StreamRole stream_role) {
  std::shared_ptr<Stream> stream(new Stream(device, stream_role));
  return JUST(GlobalMaybe<StreamMgr>())
      ->AddStreamSymbol(*stream, [&](size_t unique_stream_id) -> Maybe<Symbol<Stream>> {
        JUST(stream->Init(unique_stream_id));
        return SymbolOf(*stream);
      });
}

/*static*/ Maybe<Symbol<Stream>> Stream::New(Symbol<Device> device, StreamRole stream_role) {
  constexpr auto* Make = DECORATE(&Stream::RawNew, ThreadLocal);
  return Make(device, stream_role);
}

namespace {

Maybe<Symbol<Stream>> RawGetDefaultStreamByDevice(Symbol<Device> device) {
  return Stream::New(device, StreamRole::kCompute);
}

Maybe<Symbol<Stream>> RawGetDefaultStreamByPlacement(Symbol<ParallelDesc> parallel_desc) {
  return RawGetDefaultStreamByDevice(JUST(GetTensorDevice(parallel_desc)));
}

}  // namespace

decltype(GetDefaultStreamByDevice) GetDefaultStreamByDevice =
    DECORATE(&RawGetDefaultStreamByDevice, ThreadLocal);

decltype(GetDefaultStreamByPlacement) GetDefaultStreamByPlacement =
    DECORATE(&RawGetDefaultStreamByPlacement, ThreadLocal);

}  // namespace oneflow
