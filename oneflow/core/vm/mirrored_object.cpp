#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void MirroredObjectAccess::__Init__(VmInstructionCtx* vm_instruction_ctx,
                                    MirroredObject* mirrored_object,
                                    uint64_t logical_object_id_value, bool is_const_operand) {
  set_vm_instruction_ctx(vm_instruction_ctx);
  set_mirrored_object(mirrored_object);
  set_logical_object_id_value(logical_object_id_value);
  set_is_const_operand(is_const_operand);
}

bool MirroredObject::IsFirstTwoConsumersReadOnly() {
  auto* waiting_access_list = mut_waiting_access_list();
  if (waiting_access_list->size() < 2) { return false; }
  auto* first = waiting_access_list->Begin();
  auto* second = waiting_access_list->Next(first);
  return first->is_const_operand() && second->is_const_operand();
}

void MirroredObject::TryResetCurrentAccessType() {
  if (current_access_type().has_access_type()) { return; }
  if (waiting_access_list().empty()) { return; }
  if (IsFirstTwoConsumersReadOnly()) {
    mut_current_access_type()->mutable_read_only();
    return;
  }
  const auto& vm_stram = mut_waiting_access_list()->Begin()->vm_instruction_ctx().vm_stram();
  mut_current_access_type()->mut_vpu_id_only()->CopyFrom(vm_stram.vm_stream_id());
}

MirroredObjectAccess* MirroredObject::GetFirstAllowedAccess() {
  if (waiting_access_list().empty()) { return nullptr; }
  if (current_access_type().has_read_only()) {
    auto* first = mut_waiting_access_list()->Begin();
    if (first->is_const_operand()) { return first; }
  } else if (current_access_type().has_vpu_id_only()) {
    auto* first = mut_waiting_access_list()->Begin();
    const auto& vm_stream_id = first->vm_instruction_ctx().vm_stram().vm_stream_id();
    if (current_access_type().vpu_id_only() == vm_stream_id) { return first; }
  } else {
    UNIMPLEMENTED();
  }
  return nullptr;
}

}  // namespace oneflow
