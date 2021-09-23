#include "oneflow/core/framework/stream.cpp"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/common/decorate.h"
#include "oneflow/core/common/static_global.h"

namespace oneflow {

namespace {

Maybe<Symbol<Stream>> RawNewByDefaultDevice(const std::string& stream_type_name) {
  const auto* stream_descriptor = JUST(StreamDescriptor::Find(stream_type_name));
  const auto& device_type = stream_descriptor->device_type();
  const auto& default_device = JUST(Device::New(device_type));
  return Stream::New(stream_descriptor, default_device);
}

Maybe<Symbol<Stream>> RawNewByDefaultName(Symbol<Device> device) {
  const auto& stream_descriptor = JUST(StreamDescriptor::Find(device->type()));
  return Stream::New(stream_descriptor, device);
}

Maybe<Symbol<Stream>> RawNew(const std::string& stream_type_name, Symbol<Device> device) {
  const auto& stream_descriptor = JUST(StreamDescriptor::Find(stream_type_name));
  return Stream::RawNew(stream_descriptor, device);
}

}

decltype(Stream::NewByDefaultDevice) Stream::NewByDefaultDevice = DECORATE(&RawNewByDefaultDevice, ThreadLocalCopiable);

decltype(Stream::NewByDefaultName) Stream::NewByDefaultName = DECORATE(&RawNewByDefaultName, ThreadLocal);

decltype(Stream::New) Stream::New = DECORATE(&RawNew, ThreadLocal);

Maybe<Symbol<Stream>> Stream::RawNew(const StreamDescriptor* stream_descriptor, Symbol<Device> device) {
  Stream stream(stream_descriptor, device);
  JUST(stream.Init());
  return SymbolOf(stream);
}

namespace {

Maybe<Symbol<Device>> RawGetDefaultDevice(const StreamDescriptor* stream_descriptor) {
  return Device::New(stream_descriptor->device_type());
}

static constexpr auto* GetDefaultDevice = DECORATE(&RawGetDefaultDevice, StaticGlobalCopiable);

Maybe<LocalDepObject*> RawGetLocalDepObject(const std::string& stream_type_name, Symbol<Device> device) {
  return LocalDepObject::New(*device);
}

static constexpr auto* GetLocalDepObject = DECORATE(&RawGetLocalDepObject, StaticGlobalCopiable);

Maybe<Optional<LocalDepObject*>> GetTransportLocalDepObject(
    const StreamDescriptor* stream_descriptor) {
  const auto& shared_transport_stream = stream_descriptor->shared_transport_stream_type_name();
  if (!shared_transport_stream.has_value()) { return Optional<LocalDepObject*>(); }
  const auto& default_device = JUST(GetDefaultDevice(stream_descriptor));
  const auto& dep_object = JUST(GetLocalDepObject(shared_transport_stream, default_device));
  return Optional<LocalDepObject*>(dep_object);
}

Maybe<LocalDepObject*> GetScheduleLocalDepObject(
    const StreamDescriptor* stream_descriptor, Symbol<Device> device) {
  const auto& shared_shedule_stream = stream_descriptor->shared_schedule_stream_type_name();
  return JUST(GetLocalDepObject(shared_shedule_stream, device));
}

}

Maybe<void> Stream::Init() {
  transport_local_dep_object_ = JUST(GetTransportLocalDepObject(stream_descriptor_));
  schedule_local_dep_object_ = JUST(GetScheduleLocalDepObject(stream_descriptor_, device_));
  return Maybe<void>::Ok();
}

}
