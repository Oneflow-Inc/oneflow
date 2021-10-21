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
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/test_util.h"

namespace oneflow {
namespace vm {
namespace test {

TEST(StringStreamType, init_string_object) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewSymbol", "InitStringSymbol"});
  auto vm = ObjectMsgPtr<VirtualMachine>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t symbol_id = IdUtil::NewLogicalSymbolId();
  CHECK_JUST(Global<symbol::Storage<StringSymbol>>::Get()->Add(symbol_id, "foobar"));
  list.EmplaceBack(NewInstruction("NewSymbol")->add_int64_operand(symbol_id));
  list.EmplaceBack(NewInstruction("InitStringSymbol")->add_init_symbol_operand(symbol_id));
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  auto* logical_object = vm->mut_id2logical_object()->FindPtr(IdUtil::GetTypeId(symbol_id));
  ASSERT_NE(logical_object, nullptr);
  auto* mirrored_object = logical_object->mut_global_device_id2mirrored_object()->FindPtr(0);
  ASSERT_NE(mirrored_object, nullptr);
  const auto& string_obj = CHECK_JUST(mirrored_object->rw_mutexed_object().Get<StringObject>());
  ASSERT_TRUE(string_obj.str() == "foobar");
}

}  // namespace test
}  // namespace vm
}  // namespace oneflow
