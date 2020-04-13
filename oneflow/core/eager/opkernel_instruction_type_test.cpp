#define private public

#include "oneflow/core/vm/logical_object_id.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace eager {
namespace test {

using InstructionMsgList = OBJECT_MSG_LIST(vm::InstructionMsg, instr_msg_link);

TEST(OpkernelInstructionType, new_opkernel) {
  InstructionMsgList list;
  int64_t job_desc_id = 0;
  {
    job_desc_id = vm::TestUtil::NewSymbol(&list);
    Global<vm::Storage<JobConfigProto>>::Get()->Add(job_desc_id,
                                                    std::make_shared<JobConfigProto>());
    list.EmplaceBack(
        vm::NewInstruction("InitJobDescSymbol")->add_init_const_host_operand(job_desc_id));
  }
  int64_t op_conf_id = 0;
  {
    op_conf_id = vm::TestUtil::NewSymbol(&list);
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->mutable_user_conf()->set_op_type_name("TestSource");
    Global<vm::Storage<OperatorConf>>::Get()->Add(op_conf_id, op_conf);
    list.EmplaceBack(
        vm::NewInstruction("InitOperatorConfSymbol")->add_init_const_host_operand(op_conf_id));
  }
  int64_t op_id = vm::TestUtil::NewObject(&list, "0:cpu:0");
  list.EmplaceBack(vm::NewInstruction("InitOpKernelObject")
                       ->add_const_host_operand(job_desc_id)
                       ->add_const_host_operand(op_conf_id)
                       ->add_mut_operand(op_id));
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(),
      {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol", "InitOpKernelObject"});
  auto scheduler = ObjectMsgPtr<vm::Scheduler>::New(vm_desc.Get());
  scheduler->Receive(&list);
  while (!scheduler->Empty()) {
    scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

}  // namespace test
}  // namespace eager
}  // namespace oneflow
