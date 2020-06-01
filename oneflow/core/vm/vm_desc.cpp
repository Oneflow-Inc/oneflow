#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

void SetMachineIdRange(Range* range, int64_t machine_num, int64_t this_machine_id) {
  *range = Range(this_machine_id, this_machine_id + 1);
}

}  // namespace

ObjectMsgPtr<VmDesc> MakeVmDesc(const Resource& resource, int64_t this_machine_id) {
  std::set<StreamTypeId> stream_type_ids;
  ForEachInstrTypeId([&](const InstrTypeId& instr_type_id) {
    stream_type_ids.insert(instr_type_id.stream_type_id());
  });
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(ObjectMsgPtr<VmResourceDesc>::New(resource).Get());
  SetMachineIdRange(vm_desc->mutable_machine_id_range(), resource.machine_num(), this_machine_id);
  int cnt = 0;
  for (const auto& stream_type_id : stream_type_ids) {
    const StreamType& stream_type = stream_type_id.stream_type();
    auto stream_desc = stream_type.MakeStreamDesc(resource, this_machine_id);
    if (stream_desc) {
      ++cnt;
      CHECK(vm_desc->mut_stream_type_id2desc()->Insert(stream_desc.Mutable()).second);
    }
  }
  CHECK_EQ(vm_desc->stream_type_id2desc().size(), cnt);
  return vm_desc;
}

}  // namespace vm
}  // namespace oneflow
