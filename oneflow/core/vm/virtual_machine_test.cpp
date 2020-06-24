#include <iostream>
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/common/object_msg_reflection.h"
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
