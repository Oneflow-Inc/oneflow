#ifndef ONEFLOW_CORE_VM_INSTRUCTION_ID_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_ID_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/vm_type.h"

namespace oneflow {
namespace vm {

class InstrTypeId final {
 public:
  InstrTypeId() { __Init__(); }
  InstrTypeId(const InstrTypeId& rhs) {
    stream_type_id_.CopyFrom(rhs.stream_type_id_);
    set_opcode(rhs.opcode());
    set_type(rhs.type());
  }

  ~InstrTypeId() = default;

  void __Init__() { mutable_stream_type_id()->__Init__(); }
  void __Init__(const std::type_index& stream_type_index, InterpretType interpret_type,
                InstructionOpcode opcode, VmType type) {
    mutable_stream_type_id()->__Init__(stream_type_index, interpret_type);
    set_opcode(opcode);
    set_type(type);
  }
  void clear() {
    stream_type_id_.clear();
    opcode_ = 0;
    type_ = VmType::kInvalidVmType;
  }
  void CopyFrom(const InstrTypeId& rhs) {
    stream_type_id_.CopyFrom(rhs.stream_type_id_);
    opcode_ = rhs.opcode_;
    type_ = rhs.type_;
  }
  // Getters
  const StreamTypeId& stream_type_id() const { return stream_type_id_; }
  InstructionOpcode opcode() const { return opcode_; }
  VmType type() const { return type_; }

  // Setters
  StreamTypeId* mut_stream_type_id() { return &stream_type_id_; }
  StreamTypeId* mutable_stream_type_id() { return &stream_type_id_; }
  void set_opcode(InstructionOpcode val) { opcode_ = val; }
  void set_type(VmType val) { type_ = val; }

  bool operator==(const InstrTypeId& rhs) const {
    return stream_type_id_ == rhs.stream_type_id_ && opcode_ == rhs.opcode_ && type_ == rhs.type_;
  }
  bool operator<(const InstrTypeId& rhs) const {
    if (!(stream_type_id_ == rhs.stream_type_id_)) { return stream_type_id_ < rhs.stream_type_id_; }
    if (!(opcode_ == rhs.opcode_)) { return opcode_ < rhs.opcode_; }
    return type_ < rhs.type_;
  }
  bool operator<=(const InstrTypeId& rhs) const { return *this < rhs || *this == rhs; }

 private:
  StreamTypeId stream_type_id_;
  InstructionOpcode opcode_;
  VmType type_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_ID_H_
