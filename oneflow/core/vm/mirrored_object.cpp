#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void MirroredObjectAccess::__Init__(VmInstrChain* vm_instr_chain, MirroredObject* mirrored_object,
                                    bool is_const_operand) {
  set_vm_instr_chain(vm_instr_chain);
  set_mirrored_object(mirrored_object);
  set_is_const_operand(is_const_operand);
}

}  // namespace oneflow
