#ifndef ONEFLOW_CORE_VM_VM_PROCEDURE_H_
#define ONEFLOW_CORE_VM_VM_PROCEDURE_H_

#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(Procedure);
  // fields
  OBJECT_MSG_DEFINE_STRUCT(ParallelConf, parallel_conf);
  // links
  OBJECT_MSG_DEFINE_LIST_HEAD(InstructionMsg, vm_instr_msg_link, local_vm_instr_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(InstructionMsg, vm_instr_msg_link, remote_vm_instr_list);
OBJECT_MSG_END(Procedure);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_PROCEDURE_H_
