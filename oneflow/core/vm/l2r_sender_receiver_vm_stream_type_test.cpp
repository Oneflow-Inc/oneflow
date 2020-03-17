#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_vm_stream_type.h"
#include "oneflow/core/vm/host_vm_stream_type.h"
#include "oneflow/core/vm/l2r_sender_vm_stream_type.h"
#include "oneflow/core/vm/l2r_receiver_vm_stream_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using VmInstructionMsgList = OBJECT_MSG_LIST(VmInstructionMsg, vm_instr_msg_link);

ObjectMsgPtr<VmDesc> NewVmDesc() {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  auto host_vm_stream_desc =
      ObjectMsgPtr<VmStreamDesc>::New(HostVmStreamType::kVmStreamTypeId, 1, 1, 1);
  vm_desc->mut_vm_stream_type_id2desc()->Insert(host_vm_stream_desc.Mutable());
  auto l2r_sender_vm_stream_desc =
      ObjectMsgPtr<VmStreamDesc>::New(L2RSenderVmStreamType::kVmStreamTypeId, 1, 1, 1);
  vm_desc->mut_vm_stream_type_id2desc()->Insert(l2r_sender_vm_stream_desc.Mutable());
  auto l2r_receiver_vm_stream_desc =
      ObjectMsgPtr<VmStreamDesc>::New(L2RReceiverVmStreamType::kVmStreamTypeId, 1, 1, 1);
  vm_desc->mut_vm_stream_type_id2desc()->Insert(l2r_receiver_vm_stream_desc.Mutable());
  return vm_desc;
}

ObjectMsgPtr<VmScheduler> NewTestScheduler(uint64_t symbol_value, size_t size) {
  auto vm_desc = NewVmDesc();
  auto scheduler = ObjectMsgPtr<VmScheduler>::New(vm_desc.Get());
  VmInstructionMsgList list;
  list.EmplaceBack(ControlVmStreamType().NewSymbol(symbol_value, 1));
  list.EmplaceBack(HostVmStreamType().Malloc(symbol_value, size));
  scheduler->Receive(&list);
  return scheduler;
}

TEST(L2RSenderReceiverVmStreamType, basic) {
  uint64_t logical_token = 88888888;
  uint64_t src_symbol = 9527;
  uint64_t dst_symbol = 9528;
  size_t size = 1024;
  auto scheduler0 = NewTestScheduler(src_symbol, size);
  auto scheduler1 = NewTestScheduler(dst_symbol, size);
  scheduler0->Receive(L2RSenderVmStreamType().Send(logical_token, src_symbol, size));
  scheduler1->Receive(L2RReceiverVmStreamType().Receive(logical_token, dst_symbol, size));
  while (!(scheduler0->Empty() && scheduler1->Empty())) {
    scheduler0->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(scheduler0->mut_vm_thread_list(), t) { t->TryReceiveAndRun(); }
    scheduler1->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(scheduler1->mut_vm_thread_list(), t) { t->TryReceiveAndRun(); }
  }
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
