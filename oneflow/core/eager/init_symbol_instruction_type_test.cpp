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
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/vm_desc.h"
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
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/op_conf_symbol.h"

namespace oneflow {
namespace vm {
namespace test {

using InstructionMsgList = intrusive::List<INTRUSIVE_FIELD(vm::InstructionMsg, instr_msg_hook_)>;

template<typename T, typename SerializedT>
void TestInitSymbolInstructionType(const std::string& instr_type_name) {
  auto vm_desc = intrusive::make_shared<vm::VmDesc>(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewSymbol", instr_type_name});
  auto vm = intrusive::make_shared<vm::VirtualMachine>(vm_desc.Get());
  InstructionMsgList list;
  int64_t symbol_id = vm::IdUtil::NewLogicalSymbolId();
  CHECK_JUST(Global<symbol::Storage<T>>::Get()->Add(symbol_id, SerializedT()));
  list.EmplaceBack(vm::NewInstruction("NewSymbol")->add_int64_operand(symbol_id));
  list.EmplaceBack(vm::NewInstruction(instr_type_name)->add_init_symbol_operand(symbol_id));
  CHECK_JUST(vm->Receive(&list));
  while (!vm->Empty()) {
    vm->Schedule();
    INTRUSIVE_FOR_EACH_PTR(t, vm->mut_thread_ctx_list()) { t->TryReceiveAndRun(); }
  }
  auto* logical_object = vm->mut_id2logical_object()->FindPtr(vm::IdUtil::GetTypeId(symbol_id));
  ASSERT_NE(logical_object, nullptr);
  auto* mirrored_object = logical_object->mut_global_device_id2mirrored_object()->FindPtr(0);
  ASSERT_NE(mirrored_object, nullptr);
  ASSERT_TRUE(mirrored_object->rw_mutexed_object().Has<vm::ObjectWrapper<T>>());
}

TEST(InitSymbolInstructionType, job_desc) {
#ifdef WITH_CUDA
  vm::TestResourceDescScope resource_scope(1, 1);
#else
  vm::TestResourceDescScope resource_scope(0, 1);
#endif
  TestInitSymbolInstructionType<JobDesc, JobConfigProto>("InitJobDescSymbol");
}

TEST(InitSymbolInstructionType, operator_conf) {
#ifdef WITH_CUDA
  vm::TestResourceDescScope resource_scope(1, 1);
#else
  vm::TestResourceDescScope resource_scope(0, 1);
#endif
  TestInitSymbolInstructionType<OperatorConfSymbol, OperatorConf>("InitOperatorConfSymbol");
}

}  // namespace test
}  // namespace vm
}  // namespace oneflow
