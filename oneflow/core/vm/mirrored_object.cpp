#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void MirroredObjectAccess::__Init__(VmInstruction* vm_instruction, MirroredObject* mirrored_object,
                                    bool is_const_operand) {
  set_vm_instruction(vm_instruction);
  set_mirrored_object(mirrored_object);
  set_is_const_operand(is_const_operand);
}

}  // namespace oneflow
