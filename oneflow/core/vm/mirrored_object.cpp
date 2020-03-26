#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void MirroredObjectAccess::__Init__(InstrCtx* instr_ctx, MirroredObject* mirrored_object,
                                    bool is_const_operand) {
  set_instr_ctx(instr_ctx);
  set_mirrored_object(mirrored_object);
  set_is_const_operand(is_const_operand);
  mut_mirrored_object_id()->CopyFrom(mirrored_object->mirrored_object_id());
}

void MirroredObject::__Init__(LogicalObject* logical_object, int64_t parallel_id) {
  mut_mirrored_object_id()->__Init__(logical_object->logical_object_id(), parallel_id);
  set_logical_object(logical_object);
  set_parallel_id(parallel_id);
}

}  // namespace vm
}  // namespace oneflow
