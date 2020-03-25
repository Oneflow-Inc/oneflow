#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

TEST(HostStreamType, basic) {
  auto host_stream_desc =
      ObjectMsgPtr<StreamDesc>::New(LookupInstrTypeId("Malloc").stream_type_id(), 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_stream_type_id2desc()->Insert(
      ObjectMsgPtr<StreamDesc>::New(LookupInstrTypeId("NewSymbol").stream_type_id(), 1, 1, 1)
          .Mutable());
  vm_desc->mut_stream_type_id2desc()->Insert(host_stream_desc.Mutable());
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  uint64_t symbol_value = 9527;
  list.EmplaceBack(ControlStreamType().NewSymbol(symbol_value, 1));
  list.EmplaceBack(
      NewInstruction("CudaMallocHost")->add_mut_operand(symbol_value)->add_uint64_operand(1024));
  list.EmplaceBack(NewInstruction("CudaFreeHost")->add_mut_operand(symbol_value));
  list.EmplaceBack(ControlStreamType().DeleteSymbol(symbol_value));
  scheduler->Receive(&list);
  scheduler->Schedule();
  OBJECT_MSG_LIST_FOR_EACH_PTR(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  scheduler->Schedule();
  OBJECT_MSG_LIST_FOR_EACH_PTR(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  scheduler->Schedule();
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
  auto host_stream_desc = ObjectMsgPtr<StreamDesc>::New(
      LookupInstrTypeId("Malloc").stream_type_id(), 1, parallel_num, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_stream_type_id2desc()->Insert(
      ObjectMsgPtr<StreamDesc>::New(LookupInstrTypeId("NewSymbol").stream_type_id(), 1, 1, 1)
          .Mutable());
  vm_desc->mut_stream_type_id2desc()->Insert(host_stream_desc.Mutable());
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  uint64_t symbol_value = 9527;
  list.EmplaceBack(ControlStreamType().NewSymbol(symbol_value, parallel_num));
  list.EmplaceBack(
      NewInstruction("CudaMallocHost")->add_mut_operand(symbol_value)->add_uint64_operand(1024));
  scheduler->Receive(&list);
  scheduler->Schedule();
  OBJECT_MSG_LIST_FOR_EACH(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
