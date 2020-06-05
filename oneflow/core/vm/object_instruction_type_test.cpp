#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

TEST(ControlStreamType, new_object) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject"});
  CachedObjectMsgAllocator allocator(20, 100);
  auto vm = ObjectMsgPtr<VirtualMachine>::NewFrom(&allocator, vm_desc.Get());
  InstructionMsgList list;
  TestUtil::NewObject(&list, "0:cpu:0");
  ASSERT_TRUE(vm->pending_msg_list().empty());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

TEST(ControlStreamType, delete_object) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject"});
  CachedObjectMsgAllocator allocator(20, 100);
  auto vm = ObjectMsgPtr<VirtualMachine>::NewFrom(&allocator, vm_desc.Get());
  InstructionMsgList list;
  int64_t logical_object_id = TestUtil::NewObject(&list, "0:cpu:0");
  list.EmplaceBack(
      NewInstruction("DeleteObject")->add_mut_operand(logical_object_id, AllMirroredObject()));
  ASSERT_TRUE(vm->pending_msg_list().empty());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
