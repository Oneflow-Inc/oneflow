#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/vm.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

TEST(CudaCopyH2DStreamType, basic) {
  TestResourceDescScope scope(1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(),
                                      {"NewObject", "Malloc", "CudaMalloc", "CudaCopyH2D"});
  auto vm = ObjectMsgPtr<VirtualMachine>::New(vm_desc.Get());
  InstructionMsgList list;
  std::size_t size = 1024 * 1024;
  int64_t src_object_id = TestUtil::NewObject(&list, "0:cpu:0");
  int64_t dst_object_id = TestUtil::NewObject(&list, "0:gpu:0");
  list.EmplaceBack(
      NewInstruction("CudaMallocHost")->add_mut_operand(src_object_id)->add_int64_operand(size));
  list.EmplaceBack(
      NewInstruction("CudaMalloc")->add_mut_operand(dst_object_id)->add_int64_operand(size));
  list.EmplaceBack(NewInstruction("CudaCopyH2D")
                       ->add_mut_operand(dst_object_id)
                       ->add_const_operand(src_object_id)
                       ->add_int64_operand(size));
  vm->Receive(&list);
  size_t count = 0;
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    ++count;
  }
  // std::cout << count << std::endl;
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
