#ifndef ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/vm_stream_desc.msg.h"

namespace oneflow {

class VmStream;
class ObjectMsgAllocator;
class VmInstructionStatusQuerier;
class VmInstructionPackage;

class VmStreamType {
 public:
  virtual ~VmStreamType() = default;

  virtual VmInstructionStatusQuerier* NewStatusQuerier(ObjectMsgAllocator* allocator,
                                                       int* allocated_size,
                                                       const VmStream* vm_stream) const = 0;
  virtual void Run(VmStream* vm_stream, VmInstructionPackage* vm_instr_pkg) const = 0;

 protected:
  VmStreamType() = default;
};

const VmStreamType* LookupVmStreamType(VmStreamTypeId);
void RegisterVmStreamType(VmStreamTypeId, const VmStreamType*);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_
