#define private public
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace eager {
namespace test {

using InstructionMsgList = OBJECT_MSG_LIST(vm::InstructionMsg, instr_msg_link);

template<typename T, typename SerializedT>
void TestInitSymbolInstructionType(const std::string& instr_type_name) {
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewSymbol", instr_type_name});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t symbol_id = vm::IdUtil::NewLogicalSymbolId();
  Global<vm::Storage<SerializedT>>::Get()->Add(symbol_id, std::make_shared<SerializedT>());
  list.EmplaceBack(vm::NewInstruction("NewSymbol")->add_int64_operand(symbol_id));
  list.EmplaceBack(vm::NewInstruction(instr_type_name)->add_init_symbol_operand(symbol_id));
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  auto* logical_object = vm->mut_id2logical_object()->FindPtr(vm::IdUtil::GetTypeId(symbol_id));
  ASSERT_NE(logical_object, nullptr);
  auto* mirrored_object = logical_object->mut_global_device_id2mirrored_object()->FindPtr(0);
  ASSERT_NE(mirrored_object, nullptr);
  ASSERT_TRUE(mirrored_object->rw_mutexed_object().Has<vm::ObjectWrapper<T>>());
}

TEST(InitSymbolInstructionType, job_desc) {
  vm::TestResourceDescScope resource_scope(1, 1);
  TestInitSymbolInstructionType<JobDesc, JobConfigProto>("InitJobDescSymbol");
}

TEST(InitSymbolInstructionType, operator_conf) {
  vm::TestResourceDescScope resource_scope(1, 1);
  TestInitSymbolInstructionType<OperatorConf, OperatorConf>("InitOperatorConfSymbol");
}

}  // namespace test
}  // namespace eager
}  // namespace oneflow
