#define private public
#include "oneflow/core/vm/control_vm_stream_type.h"
#include "oneflow/core/vm/nop_vm_stream_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using VmInstructionMsgList = OBJECT_MSG_LIST(VmInstructionMsg, vm_instr_msg_link);

ObjectMsgPtr<VmScheduler> NaiveNewVmScheduler(const VmDesc& vm_desc) {
  return ObjectMsgPtr<VmScheduler>::New(vm_desc);
}

std::function<ObjectMsgPtr<VmScheduler>(const VmDesc&)> CachedAllocatorNewVmScheduler() {
  auto allocator = std::make_shared<CachedObjectMsgAllocator>(20, 100);
  return [allocator](const VmDesc& vm_desc) -> ObjectMsgPtr<VmScheduler> {
    return ObjectMsgPtr<VmScheduler>::NewFrom(allocator.get(), vm_desc);
  };
}

void TestNopVmStreamTypeNoArgument(
    std::function<ObjectMsgPtr<VmScheduler>(const VmDesc&)> NewScheduler) {
  auto nop_vm_stream_desc =
      ObjectMsgPtr<VmStreamDesc>::New(NopVmStreamType::kVmStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_vm_stream_type_id2desc()->Insert(nop_vm_stream_desc.Mutable());
  auto scheduler = NewScheduler(vm_desc.Get());
  VmInstructionMsgList list;
  auto nop_vm_instr_msg = NopVmStreamType().Nop();
  list.PushBack(nop_vm_instr_msg.Mutable());
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  scheduler->Receive(&list);
  ASSERT_EQ(scheduler->pending_msg_list().size(), 1);
  scheduler->Schedule();
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  ASSERT_EQ(scheduler->waiting_vm_instr_chain_list().size(), 0);
  ASSERT_EQ(scheduler->active_vm_stream_list().size(), 1);
  auto* vm_thread = scheduler->mut_vm_thread_list()->Begin();
  ASSERT_TRUE(vm_thread != nullptr);
  auto* vm_stream = vm_thread->mut_vm_stream_list()->Begin();
  ASSERT_TRUE(vm_stream != nullptr);
  auto* vm_instr_chain = vm_stream->mut_running_chain_list()->Begin();
  ASSERT_TRUE(vm_instr_chain != nullptr);
  auto* vm_instruction = vm_instr_chain->mut_vm_instruction_list()->Begin();
  ASSERT_TRUE(vm_instruction != nullptr);
  ASSERT_EQ(vm_instruction->mut_vm_instr_msg(), nop_vm_instr_msg.Mutable());
}

TEST(NopVmStreamType, no_argument) { TestNopVmStreamTypeNoArgument(&NaiveNewVmScheduler); }

TEST(NopVmStreamType, cached_allocator_no_argument) {
  TestNopVmStreamTypeNoArgument(CachedAllocatorNewVmScheduler());
}

void TestNopVmStreamTypeOneArgument(
    std::function<ObjectMsgPtr<VmScheduler>(const VmDesc&)> NewScheduler) {
  auto nop_vm_stream_desc =
      ObjectMsgPtr<VmStreamDesc>::New(NopVmStreamType::kVmStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_vm_stream_type_id2desc()->Insert(nop_vm_stream_desc.Mutable());
  auto scheduler = NewScheduler(vm_desc.Get());
  VmInstructionMsgList list;
  uint64_t symbol_value = 9527;
  auto ctrl_vm_instr_msg = ControlVmStreamType().NewSymbol(symbol_value, 1);
  list.PushBack(ctrl_vm_instr_msg.Mutable());
  auto nop0_vm_instr_msg = NopVmStreamType().Nop();
  nop0_vm_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop0_vm_instr_msg.Mutable());
  auto nop1_vm_instr_msg = NopVmStreamType().Nop();
  nop1_vm_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop1_vm_instr_msg.Mutable());
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  scheduler->Receive(&list);
  ASSERT_EQ(scheduler->pending_msg_list().size(), 3);
  scheduler->Schedule();
  ASSERT_TRUE(scheduler->pending_msg_list().empty());
  ASSERT_EQ(scheduler->waiting_vm_instr_chain_list().size(), 1);
  ASSERT_EQ(scheduler->active_vm_stream_list().size(), 1);
  auto* vm_thread = scheduler->mut_vm_thread_list()->Begin();
  ASSERT_TRUE(vm_thread != nullptr);
  auto* vm_stream = vm_thread->mut_vm_stream_list()->Begin();
  ASSERT_TRUE(vm_stream != nullptr);
  auto* vm_instr_chain = vm_stream->mut_running_chain_list()->Begin();
  ASSERT_TRUE(vm_instr_chain != nullptr);
  ASSERT_EQ(vm_instr_chain->out_edges().size(), 1);
  auto* vm_instruction = vm_instr_chain->mut_vm_instruction_list()->Begin();
  ASSERT_TRUE(vm_instruction != nullptr);
  ASSERT_EQ(vm_instruction->mut_vm_instr_msg(), nop0_vm_instr_msg.Mutable());
  vm_instr_chain = vm_instr_chain->mut_out_edges()->Begin()->dst_vm_instr_chain();
  ASSERT_TRUE(vm_instr_chain != nullptr);
  vm_instruction = vm_instr_chain->mut_vm_instruction_list()->Begin();
  ASSERT_TRUE(vm_instruction != nullptr);
  ASSERT_EQ(vm_instruction->mut_vm_instr_msg(), nop1_vm_instr_msg.Mutable());
}

TEST(NopVmStreamType, one_argument_dispatch) {
  TestNopVmStreamTypeOneArgument(&NaiveNewVmScheduler);
}

TEST(NopVmStreamType, cached_allocator_one_argument_dispatch) {
  TestNopVmStreamTypeOneArgument(CachedAllocatorNewVmScheduler());
}

TEST(NopVmStreamType, one_argument_triger_next_chain) {
  auto nop_vm_stream_desc =
      ObjectMsgPtr<VmStreamDesc>::New(NopVmStreamType::kVmStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_vm_stream_type_id2desc()->Insert(nop_vm_stream_desc.Mutable());
  auto scheduler = NaiveNewVmScheduler(vm_desc.Get());
  VmInstructionMsgList list;
  uint64_t symbol_value = 9527;
  auto ctrl_vm_instr_msg = ControlVmStreamType().NewSymbol(symbol_value, 1);
  list.PushBack(ctrl_vm_instr_msg.Mutable());
  auto nop0_vm_instr_msg = NopVmStreamType().Nop();
  nop0_vm_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop0_vm_instr_msg.Mutable());
  auto nop1_vm_instr_msg = NopVmStreamType().Nop();
  nop1_vm_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop1_vm_instr_msg.Mutable());
  scheduler->Receive(&list);
  scheduler->Schedule();
  scheduler->mut_vm_thread_list()->Begin()->ReceiveAndRun();
  scheduler->Schedule();
  ASSERT_EQ(scheduler->waiting_vm_instr_chain_list().size(), 0);
  ASSERT_EQ(scheduler->active_vm_stream_list().size(), 1);
  auto* vm_thread = scheduler->mut_vm_thread_list()->Begin();
  ASSERT_TRUE(vm_thread != nullptr);
  auto* vm_stream = vm_thread->mut_vm_stream_list()->Begin();
  ASSERT_TRUE(vm_stream != nullptr);
  auto* vm_instr_chain = vm_stream->mut_running_chain_list()->Begin();
  ASSERT_TRUE(vm_instr_chain != nullptr);
  ASSERT_EQ(vm_instr_chain->out_edges().size(), 0);
  auto* vm_instruction = vm_instr_chain->mut_vm_instruction_list()->Begin();
  ASSERT_TRUE(vm_instruction != nullptr);
  ASSERT_EQ(vm_instruction->mut_vm_instr_msg(), nop1_vm_instr_msg.Mutable());
}

TEST(NopVmStreamType, one_argument_triger_all_chains) {
  auto nop_vm_stream_desc =
      ObjectMsgPtr<VmStreamDesc>::New(NopVmStreamType::kVmStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_vm_stream_type_id2desc()->Insert(nop_vm_stream_desc.Mutable());
  auto scheduler = NaiveNewVmScheduler(vm_desc.Get());
  VmInstructionMsgList list;
  uint64_t symbol_value = 9527;
  auto ctrl_vm_instr_msg = ControlVmStreamType().NewSymbol(symbol_value, 1);
  list.PushBack(ctrl_vm_instr_msg.Mutable());
  auto nop0_vm_instr_msg = NopVmStreamType().Nop();
  nop0_vm_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop0_vm_instr_msg.Mutable());
  auto nop1_vm_instr_msg = NopVmStreamType().Nop();
  nop1_vm_instr_msg->add_operand()->mutable_mutable_operand()->mutable_operand()->__Init__(
      symbol_value);
  list.PushBack(nop1_vm_instr_msg.Mutable());
  scheduler->Receive(&list);
  scheduler->Schedule();
  scheduler->mut_vm_thread_list()->Begin()->ReceiveAndRun();
  scheduler->Schedule();
  scheduler->mut_vm_thread_list()->Begin()->ReceiveAndRun();
  scheduler->Schedule();
  ASSERT_EQ(scheduler->waiting_vm_instr_chain_list().size(), 0);
  ASSERT_EQ(scheduler->active_vm_stream_list().size(), 0);
  auto* vm_thread = scheduler->mut_vm_thread_list()->Begin();
  ASSERT_TRUE(vm_thread != nullptr);
  auto* vm_stream = vm_thread->mut_vm_stream_list()->Begin();
  ASSERT_TRUE(vm_stream != nullptr);
  auto* vm_instr_chain = vm_stream->mut_running_chain_list()->Begin();
  ASSERT_TRUE(vm_instr_chain == nullptr);
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
