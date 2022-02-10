#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/stream_is_transport.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/static_global.h"

namespace oneflow {

namespace {

intrusive::shared_ptr<LocalDepObject> RawNewTransportLocalDepObject() {
  return intrusive::make_shared<LocalDepObject>();
}

intrusive::shared_ptr<LocalDepObject> RawNewComputeDepObject(Symbol<Device>, StreamRole) {
  return intrusive::make_shared<LocalDepObject>();
}

}

Stream::Stream(Symbol<Device> device, StreamRole stream_role)
      : device_(device), stream_role_(stream_role), schedule_local_dep_object_(nullptr),
        transport_local_dep_object_(NullOpt) {
  static constexpr auto* GetComputeDep = DECORATE(&RawNewComputeDepObject, StaticGlobalCopiable);
  schedule_local_dep_object_ = GetComputeDep(device, stream_role).Mutable();
  if (SRSwitch<StreamIsTransport>(stream_role)) {
    static constexpr auto* GetTransportDep =
          DECORATE(&RawNewTransportLocalDepObject, StaticGlobalCopiable);
    transport_local_dep_object_ = GetTransportDep().Mutable();
  }
}

namespace {

Symbol<Stream> RawNewStream(Symbol<Device> device, StreamRole stream_role) {
  return SymbolOf(Stream(device, stream_role));
}

}

decltype(Stream::New) Stream::New = DECORATE(&RawNewStream, ThreadLocal);

}
