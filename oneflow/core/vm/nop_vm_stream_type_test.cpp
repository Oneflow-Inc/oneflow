#include "oneflow/core/vm/nop_vm_stream_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {

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

void TestNopVmStreamTypeWithoutArgument(
    std::function<ObjectMsgPtr<VmScheduler>(const VmDesc&)> NewScheduler) {
  auto nop_vm_stream_desc =
      ObjectMsgPtr<VmStreamDesc>::New(NopVmStreamType::kVmStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_vm_stream_type_id2desc()->Insert(nop_vm_stream_desc.Mutable());
  auto scheduler = NewScheduler(vm_desc.Get());
  VmInstructionMsgList list;
  auto nop_vm_instr_msg = NopVmStreamType().Nop();
  auto* nop_vm_instr_msg_ptr = nop_vm_instr_msg.Mutable();
  list.EmplaceBack(std::move(nop_vm_instr_msg));
  ASSERT_TRUE(scheduler->waiting_msg_list().empty());
  scheduler->Receive(&list);
  ASSERT_EQ(scheduler->waiting_msg_list().size(), 1);
  scheduler->Schedule();
  ASSERT_TRUE(scheduler->waiting_msg_list().empty());
  ASSERT_EQ(scheduler->waiting_vm_instr_chain_list().size(), 0);
  ASSERT_EQ(scheduler->active_vm_stream_list().size(), 1);
  auto* vm_thread = scheduler->mut_vm_thread_list()->Begin();
  ASSERT_TRUE(vm_thread != nullptr);
  auto* vm_stream = vm_thread->mut_vm_stream_list()->Begin();
  ASSERT_TRUE(vm_stream != nullptr);
  auto* vm_instr_chain_pkg = vm_stream->mut_running_pkg_list()->Begin();
  ASSERT_TRUE(vm_instr_chain_pkg != nullptr);
  auto* vm_instr_chain = vm_instr_chain_pkg->mut_vm_instr_chain_list()->Begin();
  ASSERT_TRUE(vm_instr_chain != nullptr);
  auto* vm_instruction = vm_instr_chain->mut_vm_instruction_list()->Begin();
  ASSERT_TRUE(vm_instruction != nullptr);
  ASSERT_EQ(vm_instruction->mut_vm_instr_msg(), nop_vm_instr_msg_ptr);
}

TEST(NopVmStreamType, no_argument) { TestNopVmStreamTypeWithoutArgument(&NaiveNewVmScheduler); }

TEST(NopVmStreamType, cached_allocator_no_argument) {
  TestNopVmStreamTypeWithoutArgument(CachedAllocatorNewVmScheduler());
}

}  // namespace

}  // namespace test

}  // namespace oneflow
