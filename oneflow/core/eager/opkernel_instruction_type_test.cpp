#define private public

#include "oneflow/core/vm/logical_object_id.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace eager {
namespace test {

using InstructionMsgList = OBJECT_MSG_LIST(vm::InstructionMsg, instr_msg_link);

int64_t NewJobDescSymbol(InstructionMsgList* list,
                         const std::shared_ptr<JobConfigProto>& job_conf) {
  int64_t job_desc_id = vm::TestUtil::NewSymbol(list);
  Global<vm::Storage<JobConfigProto>>::Get()->Add(job_desc_id, job_conf);
  list->EmplaceBack(vm::NewInstruction("InitJobDescSymbol")->add_init_symbol_operand(job_desc_id));
  return job_desc_id;
}

int64_t NewOpConfSymbol(InstructionMsgList* list, const std::shared_ptr<OperatorConf>& op_conf) {
  int64_t op_conf_id = vm::TestUtil::NewSymbol(list);
  Global<vm::Storage<OperatorConf>>::Get()->Add(op_conf_id, op_conf);
  list->EmplaceBack(
      vm::NewInstruction("InitOperatorConfSymbol")->add_init_symbol_operand(op_conf_id));
  return op_conf_id;
}

// return opkernel logical object id
int64_t InitOpKernelObject(InstructionMsgList* list,
                           const std::shared_ptr<JobConfigProto>& job_conf,
                           const std::shared_ptr<OperatorConf>& op_conf) {
  int64_t job_desc_id = NewJobDescSymbol(list, job_conf);
  int64_t op_conf_id = NewOpConfSymbol(list, op_conf);
  int64_t opkernel_id = vm::TestUtil::NewObject(list, "0:gpu:0");
  list->EmplaceBack(vm::NewInstruction("InitOpKernelObject")
                        ->add_symbol_operand(job_desc_id)
                        ->add_symbol_operand(op_conf_id)
                        ->add_mut_operand(opkernel_id));
  return opkernel_id;
}

TEST(OpkernelInstructionType, new_opkernel) {
  vm::TestResourceDescScope scope(1, 1);
  InstructionMsgList list;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->mutable_user_conf()->set_op_type_name("TestSource");
    InitOpKernelObject(&list, std::make_shared<JobConfigProto>(), op_conf);
  }
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(),
      {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol", "InitOpKernelObject"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

TEST(OpkernelInstructionType, delete_opkernel) {
  vm::TestResourceDescScope scope(1, 1);
  InstructionMsgList list;
  int64_t opkernel_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->mutable_user_conf()->set_op_type_name("TestSource");
    opkernel_id = InitOpKernelObject(&list, std::make_shared<JobConfigProto>(), op_conf);
  }
  list.EmplaceBack(vm::NewInstruction("DeleteOpKernelObject")->add_mut_operand(opkernel_id));
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(),
      {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol", "InitOpKernelObject"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

TEST(OpkernelInstructionType, call_opkernel) {
  vm::TestResourceDescScope scope(1, 1);
  InstructionMsgList list;
  int64_t opkernel_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("test_source_op_name");
    op_conf->mutable_user_conf()->set_op_type_name("TestSource");
    opkernel_id = InitOpKernelObject(&list, std::make_shared<JobConfigProto>(), op_conf);
  }
  int64_t obn_id = vm::TestUtil::NewStringSymbol(&list, "out");
  int64_t output_blob_id = vm::TestUtil::NewObject(&list, "0:gpu:0");
  list.EmplaceBack(vm::NewInstruction("CudaCallOpKernel")
                       ->add_mut_operand(opkernel_id)
                       ->add_separator()
                       ->add_separator()
                       ->add_symbol_operand(obn_id)
                       ->add_int64_operand(0)
                       ->add_mut_operand(output_blob_id)
                       ->add_separator());
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(), {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol",
                          "InitOpKernelObject", "CudaCallOpKernel"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

TEST(OpkernelInstructionType, consecutive_opkernel_calls) {
  vm::TestResourceDescScope scope(1, 1);
  InstructionMsgList list;
  int64_t in_id = vm::TestUtil::NewStringSymbol(&list, "in");
  int64_t out_id = vm::TestUtil::NewStringSymbol(&list, "out");
  int64_t test_source_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("test_source_op_name");
    op_conf->mutable_user_conf()->set_op_type_name("TestSource");
    test_source_id = InitOpKernelObject(&list, std::make_shared<JobConfigProto>(), op_conf);
  }
  int64_t x = 0;
  {
    x = vm::TestUtil::NewObject(&list, "0:gpu:0");
    list.EmplaceBack(vm::NewInstruction("CudaCallOpKernel")
                         ->add_mut_operand(test_source_id)
                         ->add_separator()
                         ->add_separator()
                         ->add_symbol_operand(out_id)
                         ->add_int64_operand(0)
                         ->add_mut_operand(x)
                         ->add_separator());
  }
  int64_t ccrelu_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("ccrelu_op_name");
    op_conf->mutable_user_conf()->set_op_type_name("ccrelu");
    ccrelu_id = InitOpKernelObject(&list, std::make_shared<JobConfigProto>(), op_conf);
  }
  int64_t y = 0;
  {
    y = vm::TestUtil::NewObject(&list, "0:gpu:0");
    list.EmplaceBack(vm::NewInstruction("CudaCallOpKernel")
                         ->add_mut_operand(ccrelu_id)
                         ->add_separator()
                         ->add_symbol_operand(in_id)
                         ->add_int64_operand(0)
                         ->add_const_operand(x)
                         ->add_separator()
                         ->add_symbol_operand(out_id)
                         ->add_int64_operand(0)
                         ->add_mut_operand(y)
                         ->add_separator());
  }
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(), {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol",
                          "InitOpKernelObject", "CudaCallOpKernel"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

TEST(OpkernelInstructionType, stateless_call_opkernel) {
  vm::TestResourceDescScope scope(1, 1);
  InstructionMsgList list;
  int64_t job_desc_id = NewJobDescSymbol(&list, std::make_shared<JobConfigProto>());
  int64_t opkernel_id = vm::TestUtil::NewObject(&list, "0:gpu:0");
  int64_t op_conf_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("test_source_op_name");
    op_conf->mutable_user_conf()->set_op_type_name("TestSource");
    op_conf_id = NewOpConfSymbol(&list, op_conf);
  }
  int64_t obn_id = vm::TestUtil::NewStringSymbol(&list, "out");
  int64_t output_blob_id = vm::TestUtil::NewObject(&list, "0:gpu:0");
  list.EmplaceBack(vm::NewInstruction("CudaStatelessCallOpKernel")
                       ->add_symbol_operand(job_desc_id)
                       ->add_symbol_operand(op_conf_id)
                       ->add_mut_operand(opkernel_id)
                       ->add_separator()
                       ->add_separator()
                       ->add_symbol_operand(obn_id)
                       ->add_int64_operand(0)
                       ->add_mut_operand(output_blob_id)
                       ->add_separator());
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(), {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol",
                          "InitOpKernelObject", "CudaCallOpKernel"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

TEST(OpkernelInstructionType, consecutive_stateless_call_opkernel) {
  vm::TestResourceDescScope scope(1, 1);
  InstructionMsgList list;
  int64_t job_desc_id = NewJobDescSymbol(&list, std::make_shared<JobConfigProto>());
  int64_t opkernel_id = vm::TestUtil::NewObject(&list, "0:gpu:0");
  int64_t out_id = vm::TestUtil::NewStringSymbol(&list, "out");
  int64_t test_source_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("test_source_op_name");
    op_conf->mutable_user_conf()->set_op_type_name("TestSource");
    test_source_id = NewOpConfSymbol(&list, op_conf);
  }
  int64_t x = vm::TestUtil::NewObject(&list, "0:gpu:0");
  list.EmplaceBack(vm::NewInstruction("CudaStatelessCallOpKernel")
                       ->add_symbol_operand(job_desc_id)
                       ->add_symbol_operand(test_source_id)
                       ->add_mut_operand(opkernel_id)
                       ->add_separator()
                       ->add_separator()
                       ->add_symbol_operand(out_id)
                       ->add_int64_operand(0)
                       ->add_mut_operand(x)
                       ->add_separator());
  int64_t in_id = vm::TestUtil::NewStringSymbol(&list, "in");
  int64_t ccrelu_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("ccrelu_op_name");
    op_conf->mutable_user_conf()->set_op_type_name("ccrelu");
    ccrelu_id = NewOpConfSymbol(&list, op_conf);
  }
  int64_t y = vm::TestUtil::NewObject(&list, "0:gpu:0");
  list.EmplaceBack(vm::NewInstruction("CudaStatelessCallOpKernel")
                       ->add_symbol_operand(job_desc_id)
                       ->add_symbol_operand(ccrelu_id)
                       ->add_mut_operand(opkernel_id)
                       ->add_separator()
                       ->add_symbol_operand(in_id)
                       ->add_int64_operand(0)
                       ->add_const_operand(x)
                       ->add_separator()
                       ->add_symbol_operand(out_id)
                       ->add_int64_operand(0)
                       ->add_mut_operand(y)
                       ->add_separator());
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(),
      {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol", "CudaStatelessCallOpKernel"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

}  // namespace test
}  // namespace eager
}  // namespace oneflow
