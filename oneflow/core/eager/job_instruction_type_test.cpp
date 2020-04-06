#define private public
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace eager {
namespace test {

using InstructionMsgList = OBJECT_MSG_LIST(vm::InstructionMsg, instr_msg_link);

template<vm::VmType vm_type>
ObjectMsgPtr<vm::Scheduler> NewTestScheduler() {
  Resource resource;
  resource.set_machine_num(1);
  resource.set_gpu_device_num(1);
  auto vm_desc = vm::MakeVmDesc<vm_type>(resource, 0);
  auto scheduler = ObjectMsgPtr<vm::Scheduler>::New(vm_desc.Get());
  return scheduler;
}

TEST(JobInstructionType, new_job) {
  auto scheduler = NewTestScheduler<vm::VmType::kRemote>();
  int64_t symbol = 9527;
  {
    Global<vm::Storage<Job>>::Get()->ClearAll();
    auto job = std::make_shared<Job>();
    job->mutable_placement()->mutable_placement_group()->Add();
    Global<vm::Storage<Job>>::Get()->Add(symbol, job);
  }
  InstructionMsgList list;
  {
    list.EmplaceBack(
        vm::NewInstruction("NewSymbol")->add_uint64_operand(1)->add_int64_operand(symbol));
    list.EmplaceBack(
        vm::NewInstruction("NewJobObject")->add_mut_operand(symbol)->add_int64_operand(0));
  }
  scheduler->Receive(&list);
  while (!scheduler->Empty()) {
    scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

TEST(JobInstructionType, delete_job) {
  auto scheduler = NewTestScheduler<vm::VmType::kRemote>();
  int64_t symbol = 9527;
  {
    Global<vm::Storage<Job>>::Get()->ClearAll();
    auto job = std::make_shared<Job>();
    job->mutable_placement()->mutable_placement_group()->Add();
    Global<vm::Storage<Job>>::Get()->Add(symbol, job);
  }
  InstructionMsgList list;
  {
    list.EmplaceBack(
        vm::NewInstruction("NewSymbol")->add_uint64_operand(1)->add_int64_operand(symbol));
    list.EmplaceBack(
        vm::NewInstruction("NewJobObject")->add_mut_operand(symbol)->add_int64_operand(0));
    list.EmplaceBack(vm::NewInstruction("DeleteJobObject")->add_mut_operand(symbol));
  }
  scheduler->Receive(&list);
  while (!scheduler->Empty()) {
    scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

}  // namespace test
}  // namespace eager
}  // namespace oneflow
