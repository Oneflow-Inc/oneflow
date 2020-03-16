#include "oneflow/core/vm/mirrored_object_id.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void MirroredObjectOperand::__Init__(const LogicalObjectId& logical_object_id,
                                     int64_t parallel_id) {
  set_logical_object_id(logical_object_id);
  set_fixed_parallel_id(parallel_id);
}

void MirroredObjectOperand::__Init__(const LogicalObjectId& logical_object_id) {
  set_logical_object_id(logical_object_id);
  mutable_mirrored_parallel_id();
}

void MirroredObjectOperand::__Init__(const MirroredObjectOperandProto& proto) {
  if (proto.has_fixed_parallel_id()) {
    set_fixed_parallel_id(fixed_parallel_id());
  } else if (proto.has_mirrored_parallel_id()) {
    set_mirrored_parallel_id(mirrored_parallel_id());
  } else if (proto.has_all_parallel_id()) {
    set_all_parallel_id(all_parallel_id());
  } else {
    UNIMPLEMENTED();
  }
}

int64_t MirroredObjectOperand::GetParallelId(int64_t parallel_id) const {
  if (has_fixed_parallel_id()) { return fixed_parallel_id(); }
  CHECK(has_mirrored_parallel_id());
  return parallel_id;
}

void MirroredObjectId::__Init__(uint64_t logical_object_id_value, int64_t parallel_id) {
  set_logical_object_id_value(logical_object_id_value);
  set_parallel_id(parallel_id);
}

void MirroredObjectId::__Init__(const MirroredObjectOperand& operand, int64_t parallel_id) {
  __Init__(operand.logical_object_id(), operand.GetParallelId(parallel_id));
}

}  // namespace oneflow
