#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

TEST(DeviceHelperStreamType, basic) {
  TestResourceDescScope scope(1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject", "CudaMalloc"});
  auto vm = ObjectMsgPtr<VirtualMachine>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t object_id = TestUtil::NewObject(&list, "0:gpu:0");
  list.EmplaceBack(
      NewInstruction("CudaMalloc")->add_mut_operand(object_id)->add_int64_operand(1024));
  list.EmplaceBack(NewInstruction("CudaFree")->add_mut_operand(object_id));
  list.EmplaceBack(NewInstruction("DeleteObject")->add_mut_operand(object_id, AllMirroredObject()));
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
