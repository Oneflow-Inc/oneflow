#include "oneflow/core/vm/stream_runtime_desc.msg.h"

namespace oneflow {
namespace vm {

void StreamRtDesc::__Init__(StreamDesc* stream_desc) {
  const StreamTypeId& stream_type_id = stream_desc->stream_type_id();
  reset_stream_desc(stream_desc);
  mutable_stream_type_id()->CopyFrom(stream_type_id);
}

const StreamType& StreamRtDesc::stream_type() const { return stream_type_id().stream_type(); }

}  // namespace vm
}  // namespace oneflow
