#include "oneflow/core/framework/stream.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {

namespace {

Symbol<Stream> RawNewStream(Symbol<Device> device, StreamRole stream_role) {
  return SymbolOf(Stream(device, stream_role));
}

}

decltype(Stream::New) Stream::New = DECORATE(&RawNewStream, ThreadLocal);

}
