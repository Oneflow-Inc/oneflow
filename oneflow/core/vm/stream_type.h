#ifndef ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_

#include <string>
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/instr_type_id.msg.h"
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

  virtual void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const = 0;

  virtual void InitInstructionStatus(const Stream& stream,
                                     InstructionStatusBuffer* status_buffer) const = 0;
  virtual void DeleteInstructionStatus(const Stream& stream,
                                       InstructionStatusBuffer* status_buffer) const = 0;
  virtual bool QueryInstructionStatusDone(const Stream& stream,
                                          const InstructionStatusBuffer& status_buffer) const = 0;
  virtual void Run(InstrChain* instr_chain) const = 0;

 protected:
  StreamType() = default;
};

const StreamType* LookupStreamType(StreamTypeId);
void RegisterStreamType(StreamTypeId, const StreamType*);
template<typename T>
void RegisterStreamType() {
  RegisterStreamType(T::kStreamTypeId, new T());
}

class InstrTypeId;

const InstrTypeId& LookupInstrTypeId(const std::string& instr_type_name);
void RegisterInstrTypeId(const std::string& instr_type_name, StreamTypeId stream_type_id,
                         InstructionOpcode opcode, VmType type);
template<typename T>
void RegisterInstrTypeId(const std::string& instr_type_name, InstructionOpcode opcode,
                         VmType type) {
  RegisterInstrTypeId(instr_type_name, T::kStreamTypeId, opcode, type);
}

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_
