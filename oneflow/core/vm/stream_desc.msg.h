#ifndef ONEFLOW_CORE_VM_VPU_DESC_MSG_H_
#define ONEFLOW_CORE_VM_VPU_DESC_MSG_H_

#include <cstring>
#include <typeindex>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/vm/stream_type_id.h"

namespace oneflow {
namespace vm {

class StreamId final {
 public:
  using self_type = StreamId;
  void __Init__() {}
  void __Init__(const StreamTypeId& stream_type_id, int64_t global_device_id) {
    stream_type_id_.CopyFrom(stream_type_id);
    global_device_id_ = global_device_id;
  }

  void CopyFrom(const StreamId& rhs) { __Init__(rhs.stream_type_id_, rhs.global_device_id_); }

  const StreamTypeId& stream_type_id() const { return stream_type_id_; }
  int64_t global_device_id() const { return global_device_id_; }

  bool operator==(const StreamId& rhs) const {
    return stream_type_id_ == rhs.stream_type_id_ && global_device_id_ == rhs.global_device_id_;
  }

  bool operator<(const StreamId& rhs) const {
    if (!(stream_type_id_ == rhs.stream_type_id_)) { return stream_type_id_ < rhs.stream_type_id_; }
    return global_device_id_ < rhs.global_device_id_;
  }
  bool operator<=(const StreamId& rhs) const { return *this < rhs || *this == rhs; }

 private:
  StreamTypeId stream_type_id_;
  int64_t global_device_id_;
};

// clang-format off
OBJECT_MSG_BEGIN(StreamDesc);
  // methods
  PUBLIC void __Init__() {}
  PUBLIC void __Init__(const StreamTypeId& stream_type_id, int32_t num_machines, int32_t num_streams_per_machine,
                       int32_t num_streams_per_thread);
  PUBLIC int32_t num_threads() const;
  PUBLIC int32_t parallel_num() const { return num_machines() * num_streams_per_machine(); }

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, num_machines);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, num_streams_per_machine);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, num_streams_per_thread);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, start_global_device_id);

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(7, StreamTypeId, stream_type_id);
OBJECT_MSG_END(StreamDesc);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_DESC_MSG_H_
