#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

TEST(CudaCopyH2DStreamType, basic) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  {
    auto* map = vm_desc->mut_stream_type_id2desc();
    map->Insert(
        ObjectMsgPtr<StreamDesc>::New(LookupInstrTypeId("NewSymbol").stream_type_id(), 1, 1, 1)
            .Mutable());
    auto host_stream_desc =
        ObjectMsgPtr<StreamDesc>::New(LookupInstrTypeId("Malloc").stream_type_id(), 1, 1, 1);
    map->Insert(host_stream_desc.Mutable());
    auto device_helper_stream_desc =
        ObjectMsgPtr<StreamDesc>::New(LookupInstrTypeId("CudaMalloc").stream_type_id(), 1, 1, 1);
    map->Insert(device_helper_stream_desc.Mutable());
    auto cuda_copy_h2d_stream_desc =
        ObjectMsgPtr<StreamDesc>::New(LookupInstrTypeId("CudaCopyH2D").stream_type_id(), 1, 1, 1);
    map->Insert(cuda_copy_h2d_stream_desc.Mutable());
  }
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  uint64_t src_symbol = 9527;
  uint64_t dst_symbol = 9528;
  std::size_t size = 1024 * 1024;
  list.EmplaceBack(
      NewInstruction("NewSymbol")->add_uint64_operand(src_symbol)->add_int64_operand(1));
  list.EmplaceBack(
      NewInstruction("NewSymbol")->add_uint64_operand(dst_symbol)->add_int64_operand(1));
  list.EmplaceBack(
      NewInstruction("CudaMallocHost")->add_mut_operand(src_symbol)->add_uint64_operand(size));
  list.EmplaceBack(
      NewInstruction("CudaMalloc")->add_mut_operand(dst_symbol)->add_uint64_operand(size));
  list.EmplaceBack(NewInstruction("CudaCopyH2D")
                       ->add_mut_operand(dst_symbol)
                       ->add_operand(src_symbol)
                       ->add_uint64_operand(size));
  scheduler->Receive(&list);
  size_t count = 0;
  while (!scheduler->Empty()) {
    scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    ++count;
  }
  // std::cout << count << std::endl;
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
