#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

template<VmType vm_type>
ObjectMsgPtr<VmDesc> MakeVmDesc(const Resource& resource, int64_t this_machine_id) {
  std::set<StreamTypeId> stream_type_ids;
  ForEachInstrTypeId([&](const InstrTypeId& instr_type_id) {
    if (instr_type_id.type() != vm_type) { return; }
    stream_type_ids.insert(instr_type_id.stream_type_id());
  });
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(ObjectMsgPtr<VmResourceDesc>::New(resource).Get());
  int cnt = 0;
  for (const auto& stream_type_id : stream_type_ids) {
    const StreamType& stream_type = stream_type_id.stream_type();
    auto stream_desc = stream_type.template MakeStreamDesc<vm_type>(resource, this_machine_id);
    if (stream_desc) {
      ++cnt;
      CHECK(vm_desc->mut_stream_type_id2desc()->Insert(stream_desc.Mutable()).second);
    }
  }
  CHECK_EQ(vm_desc->stream_type_id2desc().size(), cnt);
  return vm_desc;
}

template ObjectMsgPtr<VmDesc> MakeVmDesc<kRemote>(const Resource& resource,
                                                  int64_t this_machine_id);
template ObjectMsgPtr<VmDesc> MakeVmDesc<kLocal>(const Resource& resource, int64_t this_machine_id);

}  // namespace vm
}  // namespace oneflow
