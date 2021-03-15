/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// include sstream first to avoid some compiling error
// caused by the following trick
// reference: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65899
#include <sstream>
#define private public
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/op_node_signature.pb.h"

namespace oneflow {
namespace eager {
namespace test {

namespace {

void InitNumProcessPerNode() {
  Global<NumProcessPerNode>::New();
  Global<NumProcessPerNode>::Get()->set_value(1);
}

void DestroyNumProcessPerNode() { Global<NumProcessPerNode>::Delete(); }

}  // namespace

using InstructionMsgList = OBJECT_MSG_LIST(vm::InstructionMsg, instr_msg_link);

int64_t NewJobDescSymbol(InstructionMsgList* list,
                         const std::shared_ptr<JobConfigProto>& job_conf) {
  int64_t job_desc_id = vm::TestUtil::NewSymbol(list);
  CHECK_JUST(Global<symbol::Storage<JobDesc>>::Get()->Add(job_desc_id, *job_conf));
  list->EmplaceBack(vm::NewInstruction("InitJobDescSymbol")->add_init_symbol_operand(job_desc_id));
  return job_desc_id;
}

int64_t NewOpConfSymbol(InstructionMsgList* list, const std::shared_ptr<OperatorConf>& op_conf) {
  int64_t op_conf_id = vm::TestUtil::NewSymbol(list);
  CHECK_JUST(Global<symbol::Storage<OperatorConfSymbol>>::Get()->Add(op_conf_id, *op_conf));
  list->EmplaceBack(
      vm::NewInstruction("InitOperatorConfSymbol")->add_init_symbol_operand(op_conf_id));
  return op_conf_id;
}

int64_t NewOpNodeSignature(InstructionMsgList* list, const std::vector<std::string>& ibns,
                           const std::vector<int64_t>& parallel_desc_symbol_ids_4_ibns,
                           const std::vector<std::string>& obns,
                           const std::vector<int64_t>& parallel_desc_symbol_ids_4_obns) {
  OpNodeSignature op_node_signature;
  const auto& SetFakeLogicalBlobDesc = [&](const std::string& bn_in_op) {
    auto* blob_sig = op_node_signature.mutable_logical_blob_desc_signature();
    auto* map = blob_sig->mutable_bn_in_op2blob_desc();
    BlobDesc(Shape({10LL}), DataType::kFloat).ToProto(&(*map)[bn_in_op]);
  };
  auto* bn_in_op2parallel_desc_symbol_id =
      op_node_signature.mutable_parallel_signature()->mutable_bn_in_op2parallel_desc_symbol_id();
  auto* map = op_node_signature.mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel();
  for (int i = 0; i < ibns.size(); ++i) {
    (*map)[ibns[i]].mutable_broadcast_parallel();
    (*bn_in_op2parallel_desc_symbol_id)[ibns[i]] = parallel_desc_symbol_ids_4_ibns[i];
    SetFakeLogicalBlobDesc(ibns[i]);
  }
  for (int i = 0; i < obns.size(); ++i) {
    (*map)[obns[i]].mutable_broadcast_parallel();
    (*bn_in_op2parallel_desc_symbol_id)[obns[i]] = parallel_desc_symbol_ids_4_obns[i];
    SetFakeLogicalBlobDesc(obns[i]);
  }
  int64_t op_node_signature_id = vm::TestUtil::NewSymbol(list);
  CHECK_JUST(Global<symbol::Storage<OpNodeSignatureDesc>>::Get()->Add(op_node_signature_id,
                                                                      op_node_signature));
  list->EmplaceBack(vm::NewInstruction("InitOpNodeSignatureDescSymbol")
                        ->add_init_symbol_operand(op_node_signature_id));
  return op_node_signature_id;
}

// return opkernel logical object id
int64_t InitOpKernelObject(InstructionMsgList* list,
                           const std::shared_ptr<JobConfigProto>& job_conf,
                           const std::shared_ptr<OperatorConf>& op_conf) {
  int64_t job_desc_id = NewJobDescSymbol(list, job_conf);
  int64_t op_conf_id = NewOpConfSymbol(list, op_conf);
  int64_t parallel_desc_id = 0;
  int64_t opkernel_id = vm::TestUtil::NewObject(list, "gpu", "0:0", &parallel_desc_id);
  list->EmplaceBack(vm::NewInstruction("InitOpKernelObject")
                        ->add_parallel_desc(parallel_desc_id)
                        ->add_symbol_operand(job_desc_id)
                        ->add_symbol_operand(op_conf_id)
                        ->add_mut_operand(opkernel_id));
  return opkernel_id;
}

TEST(OpkernelInstructionType, new_opkernel) {
  InitNumProcessPerNode();
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
  DestroyNumProcessPerNode();
}

TEST(OpkernelInstructionType, delete_opkernel) {
  InitNumProcessPerNode();
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
  DestroyNumProcessPerNode();
}

TEST(OpkernelInstructionType, call_opkernel) {
  InitNumProcessPerNode();
  vm::TestResourceDescScope scope(1, 1);
  InstructionMsgList list;
  int64_t opkernel_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("test_source_op_name");
    auto* user_conf = op_conf->mutable_user_conf();
    user_conf->set_op_type_name("TestSource");
    (*user_conf->mutable_output())["out"].add_s("test_source_op_name/out_0");
    opkernel_id = InitOpKernelObject(&list, std::make_shared<JobConfigProto>(), op_conf);
  }
  int64_t obn_id = vm::TestUtil::NewStringSymbol(&list, "out_0");
  int64_t parallel_desc_id = 0;
  int64_t output_blob_id = vm::TestUtil::NewObject(&list, "gpu", "0:0", &parallel_desc_id);
  int64_t op_node_signature_id = NewOpNodeSignature(&list, {}, {}, {"out_0"}, {parallel_desc_id});
  list.EmplaceBack(vm::NewInstruction("gpu.CallOpKernel")
                       ->add_parallel_desc(parallel_desc_id)
                       ->add_mut_operand(opkernel_id)
                       ->add_symbol_operand(op_node_signature_id)
                       ->add_separator()
                       ->add_separator()
                       ->add_separator()
                       ->add_symbol_operand(obn_id)
                       ->add_mut_operand(output_blob_id)
                       ->add_separator());
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(), {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol",
                          "InitOpKernelObject", "gpu.CallOpKernel"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  DestroyNumProcessPerNode();
}

TEST(OpkernelInstructionType, consecutive_opkernel_calls) {
  InitNumProcessPerNode();
  vm::TestResourceDescScope scope(1, 1);
  InstructionMsgList list;
  int64_t in_id = vm::TestUtil::NewStringSymbol(&list, "in_0");
  int64_t out_id = vm::TestUtil::NewStringSymbol(&list, "out_0");
  int64_t tmp_buffer_id = vm::TestUtil::NewStringSymbol(&list, "tmp_buffer_0");
  int64_t test_source_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("test_source_op_name");
    auto* user_conf = op_conf->mutable_user_conf();
    user_conf->set_op_type_name("TestSource");
    (*user_conf->mutable_output())["out"].add_s("test_source_op_name/out_0");
    test_source_id = InitOpKernelObject(&list, std::make_shared<JobConfigProto>(), op_conf);
  }
  int64_t x = 0;
  int64_t x_parallel_desc_id = 0;
  {
    x = vm::TestUtil::NewObject(&list, "gpu", "0:0", &x_parallel_desc_id);
    int64_t op_node_signature_id =
        NewOpNodeSignature(&list, {}, {}, {"out_0"}, {x_parallel_desc_id});
    list.EmplaceBack(vm::NewInstruction("gpu.CallOpKernel")
                         ->add_parallel_desc(x_parallel_desc_id)
                         ->add_mut_operand(test_source_id)
                         ->add_symbol_operand(op_node_signature_id)
                         ->add_separator()
                         ->add_separator()
                         ->add_separator()
                         ->add_symbol_operand(out_id)
                         ->add_mut_operand(x)
                         ->add_separator());
  }
  int64_t ccrelu_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("ccrelu_op_name");
    auto* user_conf = op_conf->mutable_user_conf();
    user_conf->set_op_type_name("ccrelu");
    (*user_conf->mutable_input())["in"].add_s("ccrelu_op_name/in_0");
    (*user_conf->mutable_output())["out"].add_s("ccrelu_op_name/out_0");
    ccrelu_id = InitOpKernelObject(&list, std::make_shared<JobConfigProto>(), op_conf);
  }
  int64_t y = 0;
  {
    int64_t y_parallel_desc_id = 0;
    y = vm::TestUtil::NewObject(&list, "gpu", "0:0", &y_parallel_desc_id);
    int64_t tmp_buffer = vm::TestUtil::NewObject(&list, "gpu", "0:0", &y_parallel_desc_id);
    int64_t op_node_signature_id =
        NewOpNodeSignature(&list, {"in_0"}, {x_parallel_desc_id}, {"out_0", "tmp_buffer_0"},
                           {y_parallel_desc_id, y_parallel_desc_id});
    list.EmplaceBack(vm::NewInstruction("gpu.CallOpKernel")
                         ->add_parallel_desc(y_parallel_desc_id)
                         ->add_mut_operand(ccrelu_id)
                         ->add_symbol_operand(op_node_signature_id)
                         ->add_separator()
                         ->add_symbol_operand(in_id)
                         ->add_const_operand(x)
                         ->add_separator()
                         ->add_separator()
                         ->add_symbol_operand(out_id)
                         ->add_symbol_operand(tmp_buffer_id)
                         ->add_mut_operand(y)
                         ->add_mut_operand(tmp_buffer)
                         ->add_separator());
  }
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(), {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol",
                          "InitOpKernelObject", "gpu.CallOpKernel"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  DestroyNumProcessPerNode();
}

TEST(OpkernelInstructionType, stateless_call_opkernel) {
  InitNumProcessPerNode();
  vm::TestResourceDescScope scope(1, 1);
  InstructionMsgList list;
  int64_t job_desc_id = NewJobDescSymbol(&list, std::make_shared<JobConfigProto>());
  int64_t parallel_desc_id = 0;
  int64_t opkernel_id = vm::TestUtil::NewObject(&list, "gpu", "0:0", &parallel_desc_id);
  int64_t op_conf_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("test_source_op_name");
    auto* user_conf = op_conf->mutable_user_conf();
    user_conf->set_op_type_name("TestSource");
    (*user_conf->mutable_output())["out"].add_s("test_source_op_name/out_0");
    op_conf_id = NewOpConfSymbol(&list, op_conf);
  }
  int64_t op_node_signature_id = NewOpNodeSignature(&list, {}, {}, {"out_0"}, {parallel_desc_id});
  int64_t obn_id = vm::TestUtil::NewStringSymbol(&list, "out_0");
  int64_t output_blob_id = vm::TestUtil::NewObject(&list, "gpu", "0:0");
  list.EmplaceBack(vm::NewInstruction("gpu.compute.UserStatelessCallOpKernel")
                       ->add_parallel_desc(parallel_desc_id)
                       ->add_symbol_operand(job_desc_id)
                       ->add_symbol_operand(op_conf_id)
                       ->add_symbol_operand(op_node_signature_id)
                       ->add_mut_operand(opkernel_id)
                       ->add_separator()
                       ->add_separator()
                       ->add_separator()
                       ->add_symbol_operand(obn_id)
                       ->add_mut_operand(output_blob_id)
                       ->add_separator());
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(), {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol",
                          "InitOpKernelObject", "gpu.CallOpKernel"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  DestroyNumProcessPerNode();
}

TEST(OpkernelInstructionType, consecutive_stateless_call_opkernel) {
  InitNumProcessPerNode();
  vm::TestResourceDescScope scope(1, 1);
  InstructionMsgList list;
  int64_t job_desc_id = NewJobDescSymbol(&list, std::make_shared<JobConfigProto>());
  int64_t parallel_desc_id = 0;
  int64_t opkernel_id = vm::TestUtil::NewObject(&list, "gpu", "0:0", &parallel_desc_id);
  int64_t out_id = vm::TestUtil::NewStringSymbol(&list, "out_0");
  int64_t tmp_buffer_id = vm::TestUtil::NewStringSymbol(&list, "tmp_buffer_0");
  int64_t test_source_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("test_source_op_name");
    auto* user_conf = op_conf->mutable_user_conf();
    user_conf->set_op_type_name("TestSource");
    (*user_conf->mutable_output())["out"].add_s("test_source_op_name/out_0");
    test_source_id = NewOpConfSymbol(&list, op_conf);
  }
  int64_t x = vm::TestUtil::NewObject(&list, "gpu", "0:0");
  int64_t op_node_signature_id = NewOpNodeSignature(&list, {}, {}, {"out_0"}, {parallel_desc_id});
  list.EmplaceBack(vm::NewInstruction("gpu.compute.UserStatelessCallOpKernel")
                       ->add_parallel_desc(parallel_desc_id)
                       ->add_symbol_operand(job_desc_id)
                       ->add_symbol_operand(test_source_id)
                       ->add_symbol_operand(op_node_signature_id)
                       ->add_mut_operand(opkernel_id)
                       ->add_separator()
                       ->add_separator()
                       ->add_separator()
                       ->add_symbol_operand(out_id)
                       ->add_mut_operand(x)
                       ->add_separator());
  int64_t in_id = vm::TestUtil::NewStringSymbol(&list, "in_0");
  int64_t ccrelu_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("ccrelu_op_name");
    auto* user_conf = op_conf->mutable_user_conf();
    user_conf->set_op_type_name("ccrelu");
    (*user_conf->mutable_input())["in"].add_s("ccrelu_op_name/in_0");
    (*user_conf->mutable_output())["out"].add_s("ccrelu_op_name/out_0");
    ccrelu_id = NewOpConfSymbol(&list, op_conf);
  }
  int64_t y_parallel_desc_id = 0;
  int64_t y = vm::TestUtil::NewObject(&list, "gpu", "0:0", &y_parallel_desc_id);
  int64_t tmp_buffer = vm::TestUtil::NewObject(&list, "gpu", "0:0", &y_parallel_desc_id);
  op_node_signature_id =
      NewOpNodeSignature(&list, {"in_0"}, {parallel_desc_id}, {"out_0", "tmp_buffer_0"},
                         {y_parallel_desc_id, y_parallel_desc_id});
  list.EmplaceBack(vm::NewInstruction("gpu.compute.UserStatelessCallOpKernel")
                       ->add_parallel_desc(y_parallel_desc_id)
                       ->add_symbol_operand(job_desc_id)
                       ->add_symbol_operand(ccrelu_id)
                       ->add_symbol_operand(op_node_signature_id)
                       ->add_mut_operand(opkernel_id)
                       ->add_separator()
                       ->add_symbol_operand(in_id)
                       ->add_const_operand(x)
                       ->add_separator()
                       ->add_separator()
                       ->add_symbol_operand(out_id)
                       ->add_symbol_operand(tmp_buffer_id)
                       ->add_mut_operand(y)
                       ->add_mut_operand(tmp_buffer)
                       ->add_separator());
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(
      vm_desc.Mutable(), {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol",
                          "gpu.compute.UserStatelessCallOpKernel"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  DestroyNumProcessPerNode();
}

}  // namespace test
}  // namespace eager
}  // namespace oneflow
