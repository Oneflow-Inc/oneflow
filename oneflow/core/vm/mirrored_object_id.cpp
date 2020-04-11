#include "oneflow/core/vm/mirrored_object_id.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void Operand::__Init__(const LogicalObjectId& logical_object_id, int64_t global_device_id) {
  set_logical_object_id(logical_object_id);
  set_fixed_global_device_id(global_device_id);
}

void Operand::__Init__(const LogicalObjectId& logical_object_id) {
  set_logical_object_id(logical_object_id);
  mutable_current_global_device_id();
}

void Operand::__Init__(const LogicalObjectId& logical_object_id, const AllMirrored&) {
  set_logical_object_id(logical_object_id);
  mutable_all_mirrored();
}

void Operand::__Init__(const OperandProto& proto) {
  set_logical_object_id(proto.logical_object_id());
  if (proto.has_fixed_global_device_id()) {
    set_fixed_global_device_id(fixed_global_device_id());
  } else if (proto.has_current_global_device_id()) {
    mutable_current_global_device_id();
  } else if (proto.has_all_mirrored()) {
    mutable_all_mirrored();
  } else {
    UNIMPLEMENTED();
  }
}

int64_t Operand::GetGlobalDeviceId(int64_t current_global_device_id) const {
  if (has_fixed_global_device_id()) { return fixed_global_device_id(); }
  CHECK(has_current_global_device_id());
  return current_global_device_id;
}

void MirroredObjectId::__Init__(int64_t logical_object_id_value, int64_t global_device_id) {
  set_logical_object_id_value(logical_object_id_value);
  set_global_device_id(global_device_id);
}

}  // namespace vm
}  // namespace oneflow
