#define private public
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/nop_stream_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

ObjectMsgPtr<Scheduler> NaiveNewScheduler(const VmDesc& vm_desc) {
  return ObjectMsgPtr<Scheduler>::New(vm_desc);
}

std::function<ObjectMsgPtr<Scheduler>(const VmDesc&)> CachedAllocatorNewScheduler() {
  auto allocator = std::make_shared<CachedObjectMsgAllocator>(20, 100);
  return [allocator](const VmDesc& vm_desc) -> ObjectMsgPtr<Scheduler> {
    return ObjectMsgPtr<Scheduler>::NewFrom(allocator.get(), vm_desc);
  };
}

void TestNopStreamTypeNoArgument(
    std::function<ObjectMsgPtr<Scheduler>(const VmDesc&)> NewScheduler) {
  auto nop_stream_desc = ObjectMsgPtr<StreamDesc>::New(NopStreamType::kStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_stream_type_id2desc()->Insert(nop_stream_desc.Mutable());
  auto scheduler = NewScheduler(vm_desc.Get());
  InstructionMsgList list;
  auto nop_instr_msg = NopStreamType().Nop();
  list.PushBack(nop_instr_msg.Mutable());
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  scheduler->Receive(&list);
  ASSERT_EQ(scheduler->pending_msg_list().size(), 1);
  scheduler->Schedule();
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  ASSERT_EQ(scheduler->waiting_instr_chain_list().size(), 0);
  ASSERT_EQ(scheduler->active_stream_list().size(), 1);
  auto* thread_ctx = scheduler->mut_thread_ctx_list()->Begin();
  ASSERT_TRUE(thread_ctx != nullptr);
  auto* stream = thread_ctx->mut_stream_list()->Begin();
  ASSERT_TRUE(stream != nullptr);
  auto* instr_chain = stream->mut_running_chain_list()->Begin();
  ASSERT_TRUE(instr_chain != nullptr);
  auto* instruction = instr_chain->mut_instruction_list()->Begin();
  ASSERT_TRUE(instruction != nullptr);
  ASSERT_EQ(instruction->mut_instr_msg(), nop_instr_msg.Mutable());
}

TEST(NopStreamType, no_argument) { TestNopStreamTypeNoArgument(&NaiveNewScheduler); }

TEST(NopStreamType, cached_allocator_no_argument) {
  TestNopStreamTypeNoArgument(CachedAllocatorNewScheduler());
}

void TestNopStreamTypeOneArgument(
    std::function<ObjectMsgPtr<Scheduler>(const VmDesc&)> NewScheduler) {
  auto nop_stream_desc = ObjectMsgPtr<StreamDesc>::New(NopStreamType::kStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_stream_type_id2desc()->Insert(nop_stream_desc.Mutable());
  auto scheduler = NewScheduler(vm_desc.Get());
  InstructionMsgList list;
  uint64_t symbol_value = 9527;
  auto ctrl_instr_msg = ControlStreamType().NewSymbol(symbol_value, 1);
  list.PushBack(ctrl_instr_msg.Mutable());
  auto nop0_instr_msg = NopStreamType().Nop();
  nop0_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop0_instr_msg.Mutable());
  auto nop1_instr_msg = NopStreamType().Nop();
  nop1_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop1_instr_msg.Mutable());
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  scheduler->Receive(&list);
  ASSERT_EQ(scheduler->pending_msg_list().size(), 3);
  scheduler->Schedule();
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  ASSERT_EQ(scheduler->waiting_instr_chain_list().size(), 1);
  ASSERT_EQ(scheduler->active_stream_list().size(), 1);
  auto* thread_ctx = scheduler->mut_thread_ctx_list()->Begin();
  ASSERT_TRUE(thread_ctx != nullptr);
  auto* stream = thread_ctx->mut_stream_list()->Begin();
  ASSERT_TRUE(stream != nullptr);
  auto* instr_chain = stream->mut_running_chain_list()->Begin();
  ASSERT_TRUE(instr_chain != nullptr);
  ASSERT_EQ(instr_chain->out_edges().size(), 1);
  auto* instruction = instr_chain->mut_instruction_list()->Begin();
  ASSERT_TRUE(instruction != nullptr);
  ASSERT_EQ(instruction->mut_instr_msg(), nop0_instr_msg.Mutable());
  instr_chain = instr_chain->mut_out_edges()->Begin()->dst_instr_chain();
  ASSERT_TRUE(instr_chain != nullptr);
  instruction = instr_chain->mut_instruction_list()->Begin();
  ASSERT_TRUE(instruction != nullptr);
  ASSERT_EQ(instruction->mut_instr_msg(), nop1_instr_msg.Mutable());
}

TEST(NopStreamType, one_argument_dispatch) { TestNopStreamTypeOneArgument(&NaiveNewScheduler); }

TEST(NopStreamType, cached_allocator_one_argument_dispatch) {
  TestNopStreamTypeOneArgument(CachedAllocatorNewScheduler());
}

TEST(NopStreamType, one_argument_triger_next_chain) {
  auto nop_stream_desc = ObjectMsgPtr<StreamDesc>::New(NopStreamType::kStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_stream_type_id2desc()->Insert(nop_stream_desc.Mutable());
  auto scheduler = NaiveNewScheduler(vm_desc.Get());
  InstructionMsgList list;
  uint64_t symbol_value = 9527;
  auto ctrl_instr_msg = ControlStreamType().NewSymbol(symbol_value, 1);
  list.PushBack(ctrl_instr_msg.Mutable());
  auto nop0_instr_msg = NopStreamType().Nop();
  nop0_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop0_instr_msg.Mutable());
  auto nop1_instr_msg = NopStreamType().Nop();
  nop1_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop1_instr_msg.Mutable());
  scheduler->Receive(&list);
  scheduler->Schedule();
  scheduler->mut_thread_ctx_list()->Begin()->ReceiveAndRun();
  scheduler->Schedule();
  ASSERT_EQ(scheduler->waiting_instr_chain_list().size(), 0);
  ASSERT_EQ(scheduler->active_stream_list().size(), 1);
  auto* thread_ctx = scheduler->mut_thread_ctx_list()->Begin();
  ASSERT_TRUE(thread_ctx != nullptr);
  auto* stream = thread_ctx->mut_stream_list()->Begin();
  ASSERT_TRUE(stream != nullptr);
  auto* instr_chain = stream->mut_running_chain_list()->Begin();
  ASSERT_TRUE(instr_chain != nullptr);
  ASSERT_EQ(instr_chain->out_edges().size(), 0);
  auto* instruction = instr_chain->mut_instruction_list()->Begin();
  ASSERT_TRUE(instruction != nullptr);
  ASSERT_EQ(instruction->mut_instr_msg(), nop1_instr_msg.Mutable());
}

TEST(NopStreamType, one_argument_triger_all_chains) {
  auto nop_stream_desc = ObjectMsgPtr<StreamDesc>::New(NopStreamType::kStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_stream_type_id2desc()->Insert(nop_stream_desc.Mutable());
  auto scheduler = NaiveNewScheduler(vm_desc.Get());
  InstructionMsgList list;
  uint64_t symbol_value = 9527;
  auto ctrl_instr_msg = ControlStreamType().NewSymbol(symbol_value, 1);
  list.PushBack(ctrl_instr_msg.Mutable());
  auto nop0_instr_msg = NopStreamType().Nop();
  nop0_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop0_instr_msg.Mutable());
  auto nop1_instr_msg = NopStreamType().Nop();
  nop1_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop1_instr_msg.Mutable());
  scheduler->Receive(&list);
  scheduler->Schedule();
  scheduler->mut_thread_ctx_list()->Begin()->ReceiveAndRun();
  scheduler->Schedule();
  scheduler->mut_thread_ctx_list()->Begin()->ReceiveAndRun();
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

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
