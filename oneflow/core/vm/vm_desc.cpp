#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

// clang-format off
OBJECT_MSG_BEGIN(UniqueStreamTypeId);
  // methods
  PUBLIC void __Init__(const StreamTypeId& stream_type_id) {
    mutable_stream_type_id()->CopyFrom(stream_type_id);
  }
  // links
  OBJECT_MSG_DEFINE_MAP_KEY(StreamTypeId, stream_type_id);
OBJECT_MSG_END(UniqueStreamTypeId);
// clang-format on

using StreamTypeIdSet = OBJECT_MSG_MAP(UniqueStreamTypeId, stream_type_id);

}  // namespace

template<VmType vm_type>
ObjectMsgPtr<VmDesc> MakeVmDesc(const Resource& resource, int64_t this_machine_id) {
  StreamTypeIdSet stream_type_ids;
  ForEachInstrTypeId([&](const InstrTypeId& instr_type_id) {
    if (instr_type_id.type() != vm_type) { return; }
    auto stream_type_id = ObjectMsgPtr<UniqueStreamTypeId>::New(instr_type_id.stream_type_id());
    stream_type_ids.Insert(stream_type_id.Mutable());
  });
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  OBJECT_MSG_MAP_UNSAFE_FOR_EACH_PTR(&stream_type_ids, stream_type_id) {
    const StreamType* stream_type = LookupStreamType(stream_type_id->stream_type_id());
    auto stream_desc = stream_type->template MakeStreamDesc<vm_type>(resource, this_machine_id);
    CHECK(vm_desc->mut_stream_type_id2desc()->Insert(stream_desc.Mutable()).second);
  }
  return vm_desc;
}

template ObjectMsgPtr<VmDesc> MakeVmDesc<kRemote>(const Resource& resource,
                                                  int64_t this_machine_id);
template ObjectMsgPtr<VmDesc> MakeVmDesc<kLocal>(const Resource& resource, int64_t this_machine_id);

}  // namespace vm
}  // namespace oneflow
