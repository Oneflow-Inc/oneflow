#define private public
#include "oneflow/core/vm/logical_object_id.h"
#include "oneflow/core/vm/vm.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/test_util.h"

namespace oneflow {
namespace vm {
namespace test {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

TEST(StringStreamType, init_string_object) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(),
                                      {"NewConstHostSymbol", "InitStringSymbol"});
  auto vm = ObjectMsgPtr<VirtualMachine>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t symbol_id = NewConstHostLogicalObjectId();
  Global<Storage<std::string>>::Get()->Add(symbol_id, std::make_shared<std::string>("foobar"));
  list.EmplaceBack(NewInstruction("NewConstHostSymbol")->add_int64_operand(symbol_id));
  list.EmplaceBack(NewInstruction("InitStringSymbol")->add_init_symbol_operand(symbol_id));
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  auto* logical_object = vm->mut_id2logical_object()->FindPtr(GetTypeLogicalObjectId(symbol_id));
  ASSERT_NE(logical_object, nullptr);
  auto* mirrored_object = logical_object->mut_global_device_id2mirrored_object()->FindPtr(0);
  ASSERT_NE(mirrored_object, nullptr);
  ASSERT_TRUE(mirrored_object->Get<StringObject>().str() == "foobar");
}

}  // namespace test
}  // namespace vm
}  // namespace oneflow
