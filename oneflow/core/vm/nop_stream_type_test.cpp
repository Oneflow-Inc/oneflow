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
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
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

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

ObjectMsgPtr<VirtualMachine> NaiveNewVirtualMachine(const VmDesc& vm_desc) {
  return ObjectMsgPtr<VirtualMachine>::New(vm_desc);
}

std::function<ObjectMsgPtr<VirtualMachine>(const VmDesc&)> CachedAllocatorNewVirtualMachine() {
  auto allocator = std::make_shared<CachedObjectMsgAllocator>(20, 100);
  return [allocator](const VmDesc& vm_desc) -> ObjectMsgPtr<VirtualMachine> {
    return ObjectMsgPtr<VirtualMachine>::NewFrom(allocator.get(), vm_desc);
  };
}

ThreadCtx* FindNopThreadCtx(VirtualMachine* vm) {
  const StreamTypeId& nop_stream_type_id = LookupInstrTypeId("Nop").stream_type_id();
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm->mut_thread_ctx_list(), thread_ctx) {
    if (nop_stream_type_id == thread_ctx->stream_rt_desc().stream_desc().stream_type_id()) {
      return thread_ctx;
    }
  }
  return nullptr;
}

void TestNopStreamTypeNoArgument(
    std::function<ObjectMsgPtr<VirtualMachine>(const VmDesc&)> NewVirtualMachine) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"Nop"});
  auto vm = NewVirtualMachine(vm_desc.Get());
  InstructionMsgList list;
  auto nop_instr_msg = NewInstruction("Nop");
  list.PushBack(nop_instr_msg.Mutable());
  ASSERT_TRUE(vm->pending_msg_list().empty());
  vm->Receive(&list);
  ASSERT_EQ(vm->pending_msg_list().size(), 1 * 2);
  vm->Schedule();
  ASSERT_TRUE(vm->pending_msg_list().empty());
  ASSERT_EQ(vm->waiting_instruction_list().size(), 0);
  ASSERT_EQ(vm->active_stream_list().size(), 1 * 2);
  auto* thread_ctx = FindNopThreadCtx(vm.Mutable());
  ASSERT_TRUE(thread_ctx != nullptr);
  auto* stream = thread_ctx->mut_stream_list()->Begin();
  ASSERT_TRUE(stream != nullptr);
  auto* instruction = stream->mut_running_instruction_list()->Begin();
  ASSERT_TRUE(instruction != nullptr);
  ASSERT_EQ(instruction->mut_instr_msg(), nop_instr_msg.Mutable());
}

TEST(NopStreamType, no_argument) { TestNopStreamTypeNoArgument(&NaiveNewVirtualMachine); }

TEST(NopStreamType, cached_allocator_no_argument) {
  TestNopStreamTypeNoArgument(CachedAllocatorNewVirtualMachine());
}

void TestNopStreamTypeOneArgument(
    std::function<ObjectMsgPtr<VirtualMachine>(const VmDesc&)> NewVirtualMachine) {
  InitNumProcessPerNode();
  TestResourceDescScope scope(1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"Nop", "NewObject"});
  auto vm = NewVirtualMachine(vm_desc.Get());
  InstructionMsgList list;
  int64_t object_id = TestUtil::NewObject(&list, "cpu", "0:0");
  auto nop0_instr_msg = NewInstruction("Nop");
  nop0_instr_msg->add_mut_operand(object_id);
  list.PushBack(nop0_instr_msg.Mutable());
  auto nop1_instr_msg = NewInstruction("Nop");
  nop1_instr_msg->add_mut_operand(object_id);
  list.PushBack(nop1_instr_msg.Mutable());
  ASSERT_TRUE(vm->pending_msg_list().empty());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  DestroyNumProcessPerNode();
}

TEST(NopStreamType, one_argument_dispatch) {
  TestNopStreamTypeOneArgument(&NaiveNewVirtualMachine);
}

TEST(NopStreamType, cached_allocator_one_argument_dispatch) {
  TestNopStreamTypeOneArgument(CachedAllocatorNewVirtualMachine());
}

TEST(NopStreamType, one_argument_triger_next_instruction) {
  InitNumProcessPerNode();
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"Nop", "NewObject"});
  auto vm = NaiveNewVirtualMachine(vm_desc.Get());
  InstructionMsgList list;
  int64_t object_id = TestUtil::NewObject(&list, "cpu", "0:0");
  auto nop0_instr_msg = NewInstruction("Nop");
  nop0_instr_msg->add_mut_operand(object_id);
  list.PushBack(nop0_instr_msg.Mutable());
  auto nop1_instr_msg = NewInstruction("Nop");
  nop1_instr_msg->add_mut_operand(object_id);
  list.PushBack(nop1_instr_msg.Mutable());
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  DestroyNumProcessPerNode();
}

TEST(NopStreamType, one_argument_triger_all_instructions) {
  InitNumProcessPerNode();
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"Nop", "NewObject"});
  auto vm = NaiveNewVirtualMachine(vm_desc.Get());
  InstructionMsgList list;
  int64_t object_id = TestUtil::NewObject(&list, "cpu", "0:0");
  auto nop0_instr_msg = NewInstruction("Nop");
  nop0_instr_msg->add_mut_operand(object_id);
  list.PushBack(nop0_instr_msg.Mutable());
  auto nop1_instr_msg = NewInstruction("Nop");
  nop1_instr_msg->add_mut_operand(object_id);
  list.PushBack(nop1_instr_msg.Mutable());
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
