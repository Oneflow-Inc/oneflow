#include "oneflow/core/vm/vm_stream_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

HashMap<VmStreamTypeId, const VmStreamType*>* VmStreamType4VmStreamTypeId() {
  static HashMap<VmStreamTypeId, const VmStreamType*> map;
  return &map;
}

}  // namespace

const VmStreamType* LookupVmStreamType(VmStreamTypeId vm_stream_type_id) {
  const auto& map = *VmStreamType4VmStreamTypeId();
  auto iter = map.find(vm_stream_type_id);
  CHECK(iter != map.end());
  return iter->second;
}

void RegisterVmStreamType(VmStreamTypeId vm_stream_type_id, const VmStreamType* vm_stream_type) {
  CHECK(VmStreamType4VmStreamTypeId()->emplace(vm_stream_type_id, vm_stream_type).second);
}

}  // namespace oneflow
