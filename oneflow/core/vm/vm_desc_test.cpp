#define private public
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/object_msg_reflection.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

namespace test {

TEST(VmDesc, ToDot) {
  std::string dot_str = ObjectMsgListReflection<VmDesc>().ToDot("VmDesc");
  //  std::cout << std::endl;
  //  std::cout << dot_str << std::endl;
  //  std::cout << std::endl;
}

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

template<VmType vm_type>
std::string MallocInstruction();

template<>
std::string MallocInstruction<VmType::kRemote>() {
  return "Malloc";
}
template<>
std::string MallocInstruction<VmType::kLocal>() {
  return "LocalMalloc";
}

template<VmType vm_type>
ObjectMsgPtr<Scheduler> NewTestScheduler(uint64_t symbol_value, size_t size) {
  Resource resource;
  resource.set_machine_num(1);
  resource.set_gpu_device_num(1);
  auto vm_desc = MakeVmDesc<vm_type>(resource, 0);
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  list.EmplaceBack(
      NewInstruction("NewSymbol")->add_uint64_operand(symbol_value)->add_int64_operand(1));
  list.EmplaceBack(NewInstruction(MallocInstruction<vm_type>())
                       ->add_mut_operand(symbol_value)
                       ->add_uint64_operand(size));
  scheduler->Receive(&list);
  return scheduler;
}

TEST(VmDesc, basic) {
  uint64_t logical_token = 88888888;
  uint64_t src_symbol = 9527;
  uint64_t dst_symbol = 9528;
  size_t size = 1024;
  auto scheduler0 = NewTestScheduler<VmType::kLocal>(src_symbol, size);
  auto scheduler1 = NewTestScheduler<VmType::kRemote>(dst_symbol, size);
  scheduler0->Receive(NewInstruction("L2RLocalSend")
                          ->add_operand(src_symbol)
                          ->add_uint64_operand(logical_token)
                          ->add_uint64_operand(size));
  scheduler1->Receive(NewInstruction("L2RReceive")
                          ->add_mut_operand(dst_symbol)
                          ->add_uint64_operand(logical_token)
                          ->add_uint64_operand(size));
  while (!(scheduler0->Empty() && scheduler1->Empty())) {
    scheduler0->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(scheduler0->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    scheduler1->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(scheduler1->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
}

}  // namespace test

}  // namespace vm
}  // namespace oneflow
