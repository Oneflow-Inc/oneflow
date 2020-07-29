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
#include <iostream>
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/object_msg/object_msg_reflection.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

TEST(VirtualMachine, __Init__) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"Nop", "NewObject"});
  auto vm = ObjectMsgPtr<VirtualMachine>::New(vm_desc.Get());
  ASSERT_EQ(vm->thread_ctx_list().size(), 2 * 2);
  ASSERT_EQ(vm->stream_type_id2stream_rt_desc().size(), 2 * 2);
}

TEST(VirtualMachine, ToDot) {
  std::string dot_str = ObjectMsgListReflection<VirtualMachine>().ToDot("VirtualMachine");
  // std::cout << std::endl;
  // std::cout << dot_str << std::endl;
  // std::cout << std::endl;
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
