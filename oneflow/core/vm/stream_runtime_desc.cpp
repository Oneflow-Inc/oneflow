#include "oneflow/core/vm/stream_runtime_desc.msg.h"

namespace oneflow {
namespace vm {

void StreamRtDesc::__Init__(StreamDesc* stream_desc) {
  const StreamTypeId& stream_type_id = stream_desc->stream_type_id();
  const StreamType* stream_type = LookupStreamType(stream_type_id);
  set_stream_type(stream_type);
  reset_stream_desc(stream_desc);
  mutable_stream_type_id()->CopyFrom(stream_type_id);
}

}  // namespace vm
}  // namespace oneflow
