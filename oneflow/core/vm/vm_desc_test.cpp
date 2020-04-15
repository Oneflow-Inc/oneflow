#define private public
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/object_msg_reflection.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/vm/test_util.h"
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
ObjectMsgPtr<Scheduler> NewTestScheduler(int64_t* object_id, size_t size) {
  Resource resource;
  resource.set_machine_num(1);
  resource.set_gpu_device_num(1);
  auto vm_desc = MakeVmDesc<vm_type>(resource, 0);
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  *object_id = TestUtil::NewObject(&list, "0:cpu:0");
  list.EmplaceBack(NewInstruction(MallocInstruction<vm_type>())
                       ->add_mut_operand(*object_id)
                       ->add_int64_operand(size));
  scheduler->Receive(&list);
  return scheduler;
}

TEST(VmDesc, basic) {
  int64_t logical_token = 88888888;
  int64_t src_object_id = 9527;
  int64_t dst_object_id = 9528;
  size_t size = 1024;
  auto scheduler0 = NewTestScheduler<VmType::kLocal>(&src_object_id, size);
  auto scheduler1 = NewTestScheduler<VmType::kRemote>(&dst_object_id, size);
  scheduler0->Receive(NewInstruction("L2RLocalSend")
                          ->add_const_operand(src_object_id)
                          ->add_int64_operand(logical_token)
                          ->add_int64_operand(size));
  scheduler1->Receive(NewInstruction("L2RReceive")
                          ->add_mut_operand(dst_object_id)
                          ->add_int64_operand(logical_token)
                          ->add_int64_operand(size));
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
