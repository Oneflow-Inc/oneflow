#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/stream_type.h"
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

TEST(CudaCopyH2DStreamType, basic) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(),
                                      {"NewSymbol", "Malloc", "CudaMalloc", "CudaCopyH2D"});
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t src_symbol = 9527;
  int64_t dst_symbol = 9528;
  std::size_t size = 1024 * 1024;
  list.EmplaceBack(
      NewInstruction("NewSymbol")->add_int64_operand(src_symbol)->add_int64_operand(1));
  list.EmplaceBack(
      NewInstruction("NewSymbol")->add_int64_operand(dst_symbol)->add_int64_operand(1));
  list.EmplaceBack(
      NewInstruction("CudaMallocHost")->add_mut_operand(src_symbol)->add_int64_operand(size));
  list.EmplaceBack(
      NewInstruction("CudaMalloc")->add_mut_operand(dst_symbol)->add_int64_operand(size));
  list.EmplaceBack(NewInstruction("CudaCopyH2D")
                       ->add_mut_operand(dst_symbol)
                       ->add_operand(src_symbol)
                       ->add_int64_operand(size));
  scheduler->Receive(&list);
  size_t count = 0;
  while (!scheduler->Empty()) {
    scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    ++count;
  }
  // std::cout << count << std::endl;
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
