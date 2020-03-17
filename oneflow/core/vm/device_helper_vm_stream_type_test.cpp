#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_vm_stream_type.h"
#include "oneflow/core/vm/device_helper_vm_stream_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using VmInstructionMsgList = OBJECT_MSG_LIST(VmInstructionMsg, vm_instr_msg_link);

TEST(DeviceHelperVmStreamType, basic) {
  auto device_helper_vm_stream_desc =
      ObjectMsgPtr<VmStreamDesc>::New(DeviceHelperVmStreamType::kVmStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_vm_stream_type_id2desc()->Insert(device_helper_vm_stream_desc.Mutable());
  auto scheduler = ObjectMsgPtr<VmScheduler>::New(vm_desc.Get());
  VmInstructionMsgList list;
  uint64_t symbol_value = 9527;
  list.EmplaceBack(ControlVmStreamType().NewSymbol(symbol_value, 1));
  list.EmplaceBack(DeviceHelperVmStreamType().CudaMalloc(symbol_value, 1024));
  list.EmplaceBack(DeviceHelperVmStreamType().CudaFree(symbol_value));
  list.EmplaceBack(ControlVmStreamType().DeleteSymbol(symbol_value));
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
