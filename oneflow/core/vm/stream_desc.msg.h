#ifndef ONEFLOW_CORE_VM_VPU_DESC_MSG_H_
#define ONEFLOW_CORE_VM_VPU_DESC_MSG_H_

#include <cstring>
#include <typeindex>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/layout_standardize.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/logical_object_id.msg.h"
#include "oneflow/core/vm/vm_type.h"

namespace oneflow {
namespace vm {

using InstructionOpcode = int32_t;

class StreamTypeId final {
 public:
  void clear() { interpret_type_ = InterpretType::kInvalidInterpretType; }
  void __Init__() { stream_type_index_.__Init__(std::type_index(typeid(void))); }
  void __Init__(const std::type_index& stream_type_index, InterpretType interpret_type) {
    stream_type_index_.__Init__(stream_type_index);
    interpret_type_ = interpret_type;
  }
  void __Init__(const std::type_index& stream_type_index) {
    __Init__(stream_type_index, InterpretType::kCompute);
  }
  void __Delete__() { stream_type_index_.__Delete__(); }
  void CopyFrom(const StreamTypeId& rhs) {
    __Init__(rhs.stream_type_index_.Get(), rhs.interpret_type_);
  }

  const std::type_index& stream_type_index() const { return stream_type_index_.Get(); }
  InterpretType interpret_type() const { return interpret_type_; }

  bool operator==(const StreamTypeId& rhs) const {
    return stream_type_index_.Get() == rhs.stream_type_index_.Get()
           && interpret_type_ == rhs.interpret_type_;
  }
  bool operator<(const StreamTypeId& rhs) const {
    if (!(stream_type_index_.Get() == rhs.stream_type_index_.Get())) {
      return stream_type_index_.Get() < rhs.stream_type_index_.Get();
    }
    return interpret_type_ < rhs.interpret_type_;
  }
  bool operator<=(const StreamTypeId& rhs) const { return *this < rhs || *this == rhs; }

 private:
  LayoutStandardize<std::type_index> stream_type_index_;
  InterpretType interpret_type_;
};

// clang-format off
FLAT_MSG_BEGIN(AllStreamEnabledMask);
FLAT_MSG_END(AllStreamEnabledMask);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(StreamMask);
  FLAT_MSG_DEFINE_ONEOF(mask_type,
    FLAT_MSG_ONEOF_FIELD(AllStreamEnabledMask, all_stream_enabled)
    FLAT_MSG_ONEOF_FIELD(LogicalObjectId, enabled_parallel_desc_symbol));
FLAT_MSG_END(StreamMask);
// clang-format on

class StreamId final {
 public:
  using self_type = StreamId;
  void __Init__() {}
  void __Init__(const StreamTypeId& stream_type_id, int64_t parallel_id) {
    stream_type_id_.CopyFrom(stream_type_id);
    parallel_id_ = parallel_id;
  }

  void CopyFrom(const StreamId& rhs) { __Init__(rhs.stream_type_id_, rhs.parallel_id_); }

  const StreamTypeId& stream_type_id() const { return stream_type_id_; }
  int64_t parallel_id() const { return parallel_id_; }

  bool operator==(const StreamId& rhs) const {
    return stream_type_id_ == rhs.stream_type_id_ && parallel_id_ == rhs.parallel_id_;
  }

  bool operator<(const StreamId& rhs) const {
    if (!(stream_type_id_ == rhs.stream_type_id_)) { return stream_type_id_ < rhs.stream_type_id_; }
    return parallel_id_ < rhs.parallel_id_;
  }
  bool operator<=(const StreamId& rhs) const { return *this < rhs || *this == rhs; }

 private:
  StreamTypeId stream_type_id_;
  int64_t parallel_id_;
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
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, start_parallel_id);

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(7, StreamTypeId, stream_type_id);
OBJECT_MSG_END(StreamDesc);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_DESC_MSG_H_
