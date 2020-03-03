#ifndef ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/vm_stream_desc.msg.h"

namespace oneflow {

class VmStream;
class ObjectMsgAllocator;
class VmInstructionStatusBuffer;
class VmInstrChainPackage;

class VmStreamType {
 public:
  virtual ~VmStreamType() = default;

  virtual void InitVmInstructionStatus(const VmStream& vm_stream,
                                       VmInstructionStatusBuffer* status_buffer) const = 0;
  virtual void DeleteVmInstructionStatus(const VmStream& vm_stream,
                                         VmInstructionStatusBuffer* status_buffer) const = 0;
  virtual bool QueryVmInstructionStatusDone(
      const VmStream& vm_stream, const VmInstructionStatusBuffer& status_buffer) const = 0;
  virtual void Run(VmStream* vm_stream, VmInstrChainPackage* vm_instr_chain_pkg) const = 0;

 protected:
  VmStreamType() = default;
};

const VmStreamType* LookupVmStreamType(VmStreamTypeId);
void RegisterVmStreamType(VmStreamTypeId, const VmStreamType*);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_
