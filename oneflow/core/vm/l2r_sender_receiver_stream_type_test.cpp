#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

ObjectMsgPtr<VmDesc> NewVmDesc() {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"Malloc", "L2RSend", "L2RReceive"});
  return vm_desc;
}

ObjectMsgPtr<Scheduler> NewTestScheduler(int64_t* object_id, size_t size) {
  auto vm_desc = NewVmDesc();
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject"});
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  *object_id = TestUtil::NewObject(&list, "0:cpu:0");
  list.EmplaceBack(NewInstruction("Malloc")->add_mut_operand(*object_id)->add_int64_operand(size));
  scheduler->Receive(&list);
  return scheduler;
}

TEST(L2RSenderReceiverStreamType, basic) {
  int64_t logical_token = 88888888;
  int64_t src_object_id = 0;
  int64_t dst_object_id = 0;
  size_t size = 1024;
  auto scheduler0 = NewTestScheduler(&src_object_id, size);
  auto scheduler1 = NewTestScheduler(&dst_object_id, size);
  scheduler0->Receive(NewInstruction("L2RSend")
                          ->add_const_operand(src_object_id)
                          ->add_int64_operand(logical_token)
                          ->add_int64_operand(size));
  scheduler1->Receive(NewInstruction("L2RReceive")
                          ->add_mut_operand(dst_object_id)
                          ->add_int64_operand(logical_token)
                          ->add_int64_operand(size));
  while (!(scheduler0->Empty() && scheduler1->Empty())) {
    scheduler0->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(scheduler0->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    scheduler1->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(scheduler1->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
