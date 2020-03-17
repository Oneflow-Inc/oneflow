#ifndef ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_

#include <string>
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/instruction_id.msg.h"
#include "oneflow/core/common/callback.msg.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

class ObjectMsgAllocator;

namespace vm {

class Stream;
class InstructionStatusBuffer;
class InstrChain;

class StreamType {
 public:
  virtual ~StreamType() = default;

  virtual void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* vm_stream) const = 0;

  virtual void InitInstructionStatus(const Stream& vm_stream,
                                     InstructionStatusBuffer* status_buffer) const = 0;
  virtual void DeleteInstructionStatus(const Stream& vm_stream,
                                       InstructionStatusBuffer* status_buffer) const = 0;
  virtual bool QueryInstructionStatusDone(const Stream& vm_stream,
                                          const InstructionStatusBuffer& status_buffer) const = 0;
  virtual void Run(InstrChain* vm_instr_chain) const = 0;

 protected:
  StreamType() = default;
};

const StreamType* LookupStreamType(StreamTypeId);
void RegisterStreamType(StreamTypeId, const StreamType*);
template<typename T>
void RegisterStreamType() {
  RegisterStreamType(T::kStreamTypeId, new T());
}

class InstructionId;

const InstructionId& LookupInstructionId(const std::string& instr_type_name);
void RegisterInstructionId(const std::string& instr_type_name, StreamTypeId vm_stream_type_id,
                           InstructionOpcode opcode, VmType vm_type);
template<typename T>
void RegisterInstructionId(const std::string& instr_type_name, InstructionOpcode opcode,
                           VmType vm_type) {
  RegisterInstructionId(instr_type_name, T::kStreamTypeId, opcode, vm_type);
}

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_
