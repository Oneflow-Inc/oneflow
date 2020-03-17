#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

TEST(DeviceHelperStreamType, basic) {
  auto device_helper_stream_desc =
      ObjectMsgPtr<StreamDesc>::New(DeviceHelperStreamType::kStreamTypeId, 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_stream_type_id2desc()->Insert(device_helper_stream_desc.Mutable());
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  uint64_t symbol_value = 9527;
  list.EmplaceBack(ControlStreamType().NewSymbol(symbol_value, 1));
  list.EmplaceBack(DeviceHelperStreamType().CudaMalloc(symbol_value, 1024));
  list.EmplaceBack(DeviceHelperStreamType().CudaFree(symbol_value));
  list.EmplaceBack(ControlStreamType().DeleteSymbol(symbol_value));
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
