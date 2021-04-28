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
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

void InitNumProcessPerNode() {
  Global<NumProcessPerNode>::New();
  Global<NumProcessPerNode>::Get()->set_value(1);
}

void DestroyNumProcessPerNode() { Global<NumProcessPerNode>::Delete(); }

TEST(ControlStreamType, new_object) {
  InitNumProcessPerNode();
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject"});
  CachedObjectMsgAllocator allocator(20, 100);
  auto vm = ObjectMsgPtr<VirtualMachine>::NewFrom(&allocator, vm_desc.Get());
  InstructionMsgList list;
  TestUtil::NewObject(&list, "cpu", "0:0");
  ASSERT_TRUE(vm->pending_msg_list().empty());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  DestroyNumProcessPerNode();
}

TEST(ControlStreamType, delete_object) {
  InitNumProcessPerNode();
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject"});
  CachedObjectMsgAllocator allocator(20, 100);
  auto vm = ObjectMsgPtr<VirtualMachine>::NewFrom(&allocator, vm_desc.Get());
  InstructionMsgList list;
  int64_t logical_object_id = TestUtil::NewObject(&list, "cpu", "0:0");
  list.EmplaceBack(NewInstruction("DeleteObject")->add_del_operand(logical_object_id));
  ASSERT_TRUE(vm->pending_msg_list().empty());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  DestroyNumProcessPerNode();
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
