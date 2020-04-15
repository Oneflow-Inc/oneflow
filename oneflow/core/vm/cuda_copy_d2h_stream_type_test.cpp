#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

void TestSimple(int64_t parallel_num) {
  TestResourceDescScope scope(parallel_num, parallel_num);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"NewObject"});
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), parallel_num,
                                      {"Malloc", "CudaMalloc", "CudaCopyD2H"});
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  std::size_t size = 1024 * 1024;
  std::string last_device_id = std::to_string(parallel_num - 1);
  int64_t src_object_id = TestUtil::NewObject(&list, std::string("0:gpu:0-") + last_device_id);
  int64_t dst_object_id = TestUtil::NewObject(&list, std::string("0:cpu:0-") + last_device_id);
  list.EmplaceBack(
      NewInstruction("CudaMalloc")->add_mut_operand(src_object_id)->add_int64_operand(size));
  list.EmplaceBack(
      NewInstruction("CudaMallocHost")->add_mut_operand(dst_object_id)->add_int64_operand(size));
  list.EmplaceBack(NewInstruction("CudaCopyD2H")
                       ->add_mut_operand(dst_object_id)
                       ->add_const_operand(src_object_id)
                       ->add_int64_operand(size));
  scheduler->Receive(&list);
  size_t count = 0;
  while (!scheduler->Empty()) {
    scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    ++count;
    if (count % 1000 == 0) { std::cout << count << std::endl; }
  }
}

TEST(CudaCopyD2HStreamType, basic) { TestSimple(1); }

TEST(CudaCopyD2HStreamType, two_device) { TestSimple(2); }

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
