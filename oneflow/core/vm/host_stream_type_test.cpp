#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

TEST(HostStreamType, basic) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject", "Malloc"});
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t object_id = TestUtil::NewObject(&list, "0:cpu:0");
  list.EmplaceBack(
      NewInstruction("CudaMallocHost")->add_mut_operand(object_id)->add_int64_operand(1024));
  list.EmplaceBack(NewInstruction("CudaFreeHost")->add_mut_operand(object_id));
  list.EmplaceBack(NewInstruction("DeleteObject")->add_mut_operand(object_id, AllMirroredObject()));
  scheduler->Receive(&list);
  while (!scheduler->Empty()) {
    scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  ASSERT_EQ(scheduler->waiting_instr_chain_list().size(), 0);
  ASSERT_EQ(scheduler->active_stream_list().size(), 0);
  auto* thread_ctx = scheduler->mut_thread_ctx_list()->Begin();
  ASSERT_TRUE(thread_ctx != nullptr);
  auto* stream = thread_ctx->mut_stream_list()->Begin();
  ASSERT_TRUE(stream != nullptr);
  auto* instr_chain = stream->mut_running_chain_list()->Begin();
  ASSERT_TRUE(instr_chain == nullptr);
}

TEST(HostStreamType, two_device) {
  int64_t parallel_num = 2;
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject"});
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), parallel_num, {"Malloc"});
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t object_id = TestUtil::NewObject(&list, "0:cpu:0-1");
  list.EmplaceBack(
      NewInstruction("CudaMallocHost")->add_mut_operand(object_id)->add_int64_operand(1024));
  scheduler->Receive(&list);
  while (!scheduler->Empty()) {
    scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
