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
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/object_msg/object_msg_reflection.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

namespace test {

TEST(VmDesc, ToDot) {
  std::string dot_str = ObjectMsgListReflection<VmDesc>().ToDot("VmDesc");
  // std::cout << std::endl;
  // std::cout << dot_str << std::endl;
  // std::cout << std::endl;
}

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

ObjectMsgPtr<VirtualMachine> NewTestVirtualMachine(int64_t* object_id, size_t size) {
  Resource resource;
  resource.set_machine_num(1);
  resource.set_gpu_device_num(1);
  auto vm_desc = MakeVmDesc(resource, 0);
  auto vm = ObjectMsgPtr<VirtualMachine>::New(vm_desc.Get());
  InstructionMsgList list;
  *object_id = TestUtil::NewObject(&list, "cpu", "0:0");
  list.EmplaceBack(NewInstruction("Malloc")->add_mut_operand(*object_id)->add_int64_operand(size));
  vm->Receive(&list);
  return vm;
}

TEST(VmDesc, basic) {
  int64_t logical_token = 88888888;
  int64_t src_object_id = 9527;
  int64_t dst_object_id = 9528;
  size_t size = 1024;
  auto vm0 = NewTestVirtualMachine(&src_object_id, size);
  auto vm1 = NewTestVirtualMachine(&dst_object_id, size);
  vm0->Receive(NewInstruction("L2RSend")
                   ->add_const_operand(src_object_id)
                   ->add_int64_operand(logical_token)
                   ->add_int64_operand(size));
  vm1->Receive(NewInstruction("L2RReceive")
                   ->add_mut_operand(dst_object_id)
                   ->add_int64_operand(logical_token)
                   ->add_int64_operand(size));
  while (!(vm0->Empty() && vm1->Empty())) {
    vm0->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(vm0->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    vm1->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(vm1->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

}  // namespace test

}  // namespace vm
}  // namespace oneflow
