#define private public
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

TEST(ControlStreamType, new_symbol) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_stream_type_id2desc()->Insert(
      ObjectMsgPtr<StreamDesc>::New(ControlStreamType::kStreamTypeId, 1, 1, 1).Mutable());
  CachedObjectMsgAllocator allocator(20, 100);
  auto scheduler = ObjectMsgPtr<Scheduler>::NewFrom(&allocator, vm_desc.Get());
  InstructionMsgList list;
  int64_t parallel_num = 8;
  uint64_t symbol_value = 9527;
  list.EmplaceBack(ControlStreamType().NewSymbol(symbol_value, parallel_num));
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  scheduler->Receive(&list);
  ASSERT_EQ(scheduler->pending_msg_list().size(), 1);
  scheduler->Schedule();
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  ASSERT_TRUE(scheduler->waiting_instr_chain_list().empty());
  ASSERT_TRUE(scheduler->active_stream_list().empty());
  ASSERT_EQ(scheduler->thread_ctx_list().size(), 1);
  ASSERT_EQ(scheduler->stream_type_id2stream_rt_desc().size(), 1);
  ASSERT_EQ(scheduler->id2logical_object().size(), 1);
  auto* logical_object = scheduler->mut_id2logical_object()->FindPtr(symbol_value);
  ASSERT_NE(logical_object, nullptr);
  ASSERT_EQ(logical_object->parallel_id2mirrored_object().size(), parallel_num);
  ASSERT_TRUE(scheduler->Empty());
}

TEST(ControlStreamType, delete_symbol) {
  auto nop_stream_desc =
      ObjectMsgPtr<StreamDesc>::New(LookupInstrTypeId("Nop").stream_type_id(), 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_stream_type_id2desc()->Insert(
      ObjectMsgPtr<StreamDesc>::New(ControlStreamType::kStreamTypeId, 1, 1, 1).Mutable());
  vm_desc->mut_stream_type_id2desc()->Insert(nop_stream_desc.Mutable());
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t parallel_num = 8;
  uint64_t symbol_value = 9527;
  list.EmplaceBack(ControlStreamType().NewSymbol(symbol_value, parallel_num));
  auto nop0_instr_msg = NewInstruction("Nop");
  nop0_instr_msg->add_mut_operand(symbol_value);
  list.PushBack(nop0_instr_msg.Mutable());
  list.EmplaceBack(ControlStreamType().DeleteSymbol(symbol_value));
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  scheduler->Receive(&list);
  ASSERT_EQ(scheduler->pending_msg_list().size(), 3);
  scheduler->Schedule();
  OBJECT_MSG_LIST_FOR_EACH_PTR(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  scheduler->Schedule();
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  ASSERT_TRUE(scheduler->waiting_instr_chain_list().empty());
  ASSERT_TRUE(scheduler->active_stream_list().empty());
  ASSERT_EQ(scheduler->thread_ctx_list().size(), 2);
  ASSERT_EQ(scheduler->stream_type_id2stream_rt_desc().size(), 2);
  ASSERT_TRUE(scheduler->id2logical_object().empty());
  ASSERT_TRUE(scheduler->Empty());
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
