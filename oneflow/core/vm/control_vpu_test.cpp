#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/control_vpu.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

TEST(ControlVpu, new_symbol) {
  auto scheduler = ObjectMsgPtr<VpuScheduler>::New();
  VpuInstructionMsgList list;
  int64_t parallel_num = 8;
  FlatMsg<LogicalObjectId> logical_object_id;
  uint64_t symbol_value = 9527;
  logical_object_id->set_remote_value(symbol_value);
  list.EmplaceBack(ControlVpu().NewMirroredObjectSymbol(symbol_value, true, parallel_num));
  ASSERT_TRUE(scheduler->waiting_msg_list().empty());
  scheduler->Receive(&list);
  ASSERT_EQ(scheduler->waiting_msg_list().size(), 1);
  scheduler->Dispatch();
  ASSERT_TRUE(scheduler->waiting_msg_list().empty());
  ASSERT_TRUE(scheduler->tmp_waiting_msg_list().empty());
  ASSERT_TRUE(scheduler->new_vpu_instr_ctx_list().empty());
  ASSERT_TRUE(scheduler->waiting_vpu_instr_ctx_list().empty());
  ASSERT_TRUE(scheduler->ready_vpu_instr_ctx_list().empty());
  ASSERT_TRUE(scheduler->maybe_available_access_list().empty());
  ASSERT_TRUE(scheduler->active_vpu_ctx_list().empty());
  ASSERT_TRUE(scheduler->vpu_set_ctx_list().empty());
  ASSERT_TRUE(scheduler->vpu_type_id2vpu_type_ctx().empty());
  ASSERT_TRUE(scheduler->zombie_logical_object_list().empty());
  ASSERT_EQ(scheduler->id2logical_object().size(), 1);
  auto* logical_object = scheduler->mut_id2logical_object()->FindPtr(logical_object_id.Get());
  ASSERT_NE(logical_object, nullptr);
  ASSERT_EQ(logical_object->parallel_id2mirrored_object().size(), parallel_num);
}

TEST(ControlVpu, delete_symbol) {
  auto scheduler = ObjectMsgPtr<VpuScheduler>::New();
  VpuInstructionMsgList list;
  int64_t parallel_num = 8;
  FlatMsg<LogicalObjectId> logical_object_id;
  uint64_t symbol_value = 9527;
  logical_object_id->set_remote_value(symbol_value);
  list.EmplaceBack(ControlVpu().NewMirroredObjectSymbol(symbol_value, true, parallel_num));
  list.EmplaceBack(ControlVpu().DeleteMirroredObjectSymbol(logical_object_id.Get()));
  ASSERT_TRUE(scheduler->waiting_msg_list().empty());
  scheduler->Receive(&list);
  ASSERT_EQ(scheduler->waiting_msg_list().size(), 2);
  scheduler->Dispatch();
  ASSERT_TRUE(scheduler->waiting_msg_list().empty());
  ASSERT_TRUE(scheduler->tmp_waiting_msg_list().empty());
  ASSERT_TRUE(scheduler->new_vpu_instr_ctx_list().empty());
  ASSERT_TRUE(scheduler->waiting_vpu_instr_ctx_list().empty());
  ASSERT_TRUE(scheduler->ready_vpu_instr_ctx_list().empty());
  ASSERT_TRUE(scheduler->maybe_available_access_list().empty());
  ASSERT_TRUE(scheduler->active_vpu_ctx_list().empty());
  ASSERT_TRUE(scheduler->vpu_set_ctx_list().empty());
  ASSERT_TRUE(scheduler->vpu_type_id2vpu_type_ctx().empty());
  ASSERT_TRUE(scheduler->zombie_logical_object_list().empty());
  ASSERT_TRUE(scheduler->id2logical_object().empty());
}

}  // namespace

}  // namespace test

}  // namespace oneflow
