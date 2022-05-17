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
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/intrusive/intrusive.h"

namespace oneflow {

namespace {

intrusive::shared_ptr<LocalDepObject> RawGetStaticGlobalTransportLocalDepObject() {
  return intrusive::make_shared<LocalDepObject>();
}

intrusive::shared_ptr<LocalDepObject> RawNewComputeDepObject(Symbol<Device>, StreamRole) {
  return intrusive::make_shared<LocalDepObject>();
}

}  // namespace

LocalDepObject* GetStaticGlobalTransportLocalDepObject() {
  static constexpr auto* GetLocalDepObject =
      DECORATE(&RawGetStaticGlobalTransportLocalDepObject, StaticGlobalCopiable);
  return GetLocalDepObject().Mutable();
}

Stream::Stream(Symbol<Device> device, StreamRole stream_role)
    : device_(device),
      stream_role_(stream_role),
      schedule_local_dep_object_(nullptr),
      transport_local_dep_object_(NullOpt) {
  static constexpr auto* GetComputeDep = DECORATE(&RawNewComputeDepObject, StaticGlobalCopiable);
  schedule_local_dep_object_ = GetComputeDep(device, stream_role).Mutable();
  if (StreamRoleSwitch<IsCommNetStream>(stream_role)) {
    transport_local_dep_object_ = GetStaticGlobalTransportLocalDepObject();
  }
}

namespace {

Symbol<Stream> RawNewStream(Symbol<Device> device, StreamRole stream_role) {
  return SymbolOf(Stream(device, stream_role));
}

Symbol<Stream> RawGetDefaultStreamByDevice(Symbol<Device> device) {
  return Stream::New(device, StreamRole::kCompute);
}

Maybe<Symbol<Stream>> RawGetDefaultStreamByPlacement(Symbol<ParallelDesc> parallel_desc) {
  return RawGetDefaultStreamByDevice(JUST(GetTensorDevice(parallel_desc)));
}

}  // namespace

decltype(Stream::New) Stream::New = DECORATE(&RawNewStream, ThreadLocal);

decltype(GetDefaultStreamByDevice) GetDefaultStreamByDevice =
    DECORATE(&RawGetDefaultStreamByDevice, ThreadLocal);

decltype(GetDefaultStreamByPlacement) GetDefaultStreamByPlacement =
    DECORATE(&RawGetDefaultStreamByPlacement, ThreadLocal);

}  // namespace oneflow
