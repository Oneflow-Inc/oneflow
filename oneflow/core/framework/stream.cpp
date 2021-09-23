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
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/static_global.h"

namespace oneflow {

namespace {

Maybe<Symbol<Stream>> RawNewByDefaultDevice(const std::string& stream_type_name) {
  const auto* stream_descriptor = JUST(StreamDescriptor::Find(stream_type_name));
  const auto& device_type = stream_descriptor->device_type();
  const auto& default_device = JUST(Device::New(device_type));
  return Stream::RawNew(stream_descriptor, default_device);
}

Maybe<Symbol<Stream>> RawNewByDefaultName(Symbol<Device> device) {
  const auto& stream_descriptor = JUST(StreamDescriptor::Find(device->type()));
  return Stream::RawNew(stream_descriptor, device);
}

Maybe<Symbol<Stream>> RawNewStream(const std::string& stream_type_name, Symbol<Device> device) {
  const auto& stream_descriptor = JUST(StreamDescriptor::Find(stream_type_name));
  return Stream::RawNew(stream_descriptor, device);
}

}  // namespace

decltype(Stream::NewByDefaultDevice) Stream::NewByDefaultDevice =
    DECORATE(&RawNewByDefaultDevice, ThreadLocalCopiable);

decltype(Stream::NewByDefaultName) Stream::NewByDefaultName =
    DECORATE(&RawNewByDefaultName, ThreadLocal);

decltype(Stream::New) Stream::New = DECORATE(&RawNewStream, ThreadLocalCopiable);

Maybe<Symbol<Stream>> Stream::RawNew(const StreamDescriptor* stream_descriptor,
                                     Symbol<Device> device) {
  Stream stream(stream_descriptor, device);
  JUST(stream.Init());
  return SymbolOf(stream);
}

namespace {

Maybe<Symbol<Device>> RawGetDefaultDevice(const StreamDescriptor* stream_descriptor) {
  return Device::New(stream_descriptor->device_type());
}

static constexpr auto* GetDefaultDevice = DECORATE(&RawGetDefaultDevice, StaticGlobalCopiable);

Maybe<ObjectMsgPtr<LocalDepObject>> RawGetLocalDepObject(const std::string& stream_type_name,
                                                         Symbol<Device> device) {
  return LocalDepObject::New(*device);
}

static constexpr auto* GetLocalDepObject = DECORATE(&RawGetLocalDepObject, StaticGlobalCopiable);

Maybe<Optional<LocalDepObject*>> GetTransportLocalDepObject(
    const StreamDescriptor* stream_descriptor) {
  const auto& shared_transport_stream = stream_descriptor->shared_transport_stream_type_name();
  if (!shared_transport_stream.has_value()) { return Optional<LocalDepObject*>(); }
  const auto& default_device = JUST(GetDefaultDevice(stream_descriptor));
  const auto& dep_object =
      JUST(GetLocalDepObject(*JUST(shared_transport_stream), default_device))->Mutable();
  return Optional<LocalDepObject*>(dep_object);
}

Maybe<LocalDepObject*> GetScheduleLocalDepObject(const StreamDescriptor* stream_descriptor,
                                                 Symbol<Device> device) {
  const auto& shared_shedule_stream = stream_descriptor->shared_schedule_stream_type_name();
  return JUST(GetLocalDepObject(shared_shedule_stream, device))->Mutable();
}

}  // namespace

Maybe<void> Stream::Init() {
  transport_local_dep_object_ = *JUST(GetTransportLocalDepObject(stream_descriptor_));
  schedule_local_dep_object_ = JUST(GetScheduleLocalDepObject(stream_descriptor_, device_));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
