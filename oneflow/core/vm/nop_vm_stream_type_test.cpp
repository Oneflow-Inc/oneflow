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
  return;  // TODO(lixinqi)
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  auto scheduler = NewScheduler(vm_desc.Get());
  VmInstructionMsgList list;
  list.EmplaceBack(NopVmStreamType().Nop());
  ASSERT_TRUE(scheduler->waiting_msg_list().empty());
  scheduler->Receive(&list);
  ASSERT_EQ(scheduler->waiting_msg_list().size(), 1);
  scheduler->Schedule();
  ASSERT_TRUE(scheduler->waiting_msg_list().empty());
  ASSERT_EQ(scheduler->waiting_vm_instr_chain_list().size(), 1);
  ASSERT_EQ(scheduler->active_vm_stream_list().size(), 1);
}

TEST(NopVmStreamType, no_argument) { TestNopVmStreamTypeWithoutArgument(&NaiveNewVmScheduler); }

TEST(NopVmStreamType, cached_allocator_no_argument) {
  TestNopVmStreamTypeWithoutArgument(CachedAllocatorNewVmScheduler());
}

}  // namespace

}  // namespace test

}  // namespace oneflow
