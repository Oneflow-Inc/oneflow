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
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

TEST(HostStreamType, basic) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject", "Malloc"});
  auto vm = ObjectMsgPtr<VirtualMachine>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t object_id = TestUtil::NewObject(&list, "cpu", "0:0");
  list.EmplaceBack(
      NewInstruction("CudaMallocHost")->add_mut_operand(object_id)->add_int64_operand(1024));
  list.EmplaceBack(NewInstruction("CudaFreeHost")->add_mut_operand(object_id));
  list.EmplaceBack(NewInstruction("DeleteObject")->add_mut_operand(object_id, AllMirroredObject()));
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  ASSERT_EQ(vm->waiting_instruction_list().size(), 0);
  ASSERT_EQ(vm->active_stream_list().size(), 0);
  auto* thread_ctx = vm->mut_thread_ctx_list()->Begin();
  ASSERT_TRUE(thread_ctx != nullptr);
  auto* stream = thread_ctx->mut_stream_list()->Begin();
  ASSERT_TRUE(stream != nullptr);
  auto* instruction = stream->mut_running_instruction_list()->Begin();
  ASSERT_TRUE(instruction == nullptr);
}

TEST(HostStreamType, two_device) {
  int64_t parallel_num = 2;
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject"});
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), parallel_num, {"Malloc"});
  auto vm = ObjectMsgPtr<VirtualMachine>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t object_id = TestUtil::NewObject(&list, "cpu", "0:0-1");
  list.EmplaceBack(
      NewInstruction("CudaMallocHost")->add_mut_operand(object_id)->add_int64_operand(1024));
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
