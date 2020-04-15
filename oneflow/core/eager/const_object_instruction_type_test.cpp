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
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace eager {
namespace test {

using InstructionMsgList = OBJECT_MSG_LIST(vm::InstructionMsg, instr_msg_link);

template<typename T, typename SerializedT>
void TestConstObjectInstructionType(const std::string& instr_type_name) {
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(),
                                          {"NewConstHostSymbol", instr_type_name});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  InstructionMsgList list;
  int64_t symbol_id = vm::NewConstHostLogicalObjectId();
  Global<vm::Storage<SerializedT>>::Get()->Add(symbol_id, std::make_shared<SerializedT>());
  list.EmplaceBack(vm::NewInstruction("NewConstHostSymbol")->add_int64_operand(symbol_id));
  list.EmplaceBack(vm::NewInstruction(instr_type_name)->add_init_symbol_operand(symbol_id));
  vm->Receive(&list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  auto* logical_object =
      vm->mut_id2logical_object()->FindPtr(vm::GetTypeLogicalObjectId(symbol_id));
  ASSERT_NE(logical_object, nullptr);
  auto* mirrored_object = logical_object->mut_global_device_id2mirrored_object()->FindPtr(0);
  ASSERT_NE(mirrored_object, nullptr);
  ASSERT_TRUE(mirrored_object->Has<vm::ObjectWrapper<T>>());
}

TEST(ConstObjectInstructionType, job_desc) {
  TestConstObjectInstructionType<JobDesc, JobConfigProto>("InitJobDescSymbol");
}

TEST(ConstObjectInstructionType, operator_conf) {
  TestConstObjectInstructionType<OperatorConf, OperatorConf>("InitOperatorConfSymbol");
}

}  // namespace test
}  // namespace eager
}  // namespace oneflow
