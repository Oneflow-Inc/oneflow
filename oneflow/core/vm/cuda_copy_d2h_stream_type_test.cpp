#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/cuda_copy_d2h_stream_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, vm_instr_msg_link);

void TestSimple(int64_t parallel_num) {
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  {
    auto* map = vm_desc->mut_vm_stream_type_id2desc();
    auto host_vm_stream_desc =
        ObjectMsgPtr<StreamDesc>::New(HostStreamType::kStreamTypeId, 1, parallel_num, 1);
    map->Insert(host_vm_stream_desc.Mutable());
    auto device_helper_vm_stream_desc =
        ObjectMsgPtr<StreamDesc>::New(DeviceHelperStreamType::kStreamTypeId, 1, parallel_num, 1);
    map->Insert(device_helper_vm_stream_desc.Mutable());
    auto cuda_copy_d2h_vm_stream_desc =
        ObjectMsgPtr<StreamDesc>::New(CudaCopyD2HStreamType::kStreamTypeId, 1, parallel_num, 1);
    map->Insert(cuda_copy_d2h_vm_stream_desc.Mutable());
  }
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  InstructionMsgList list;
  uint64_t src_symbol = 9527;
  uint64_t dst_symbol = 9528;
  std::size_t size = 1024 * 1024;
  list.EmplaceBack(ControlStreamType().NewSymbol(src_symbol, parallel_num));
  list.EmplaceBack(ControlStreamType().NewSymbol(dst_symbol, parallel_num));
  list.EmplaceBack(DeviceHelperStreamType().CudaMalloc(src_symbol, size));
  list.EmplaceBack(HostStreamType().CudaMallocHost(dst_symbol, size));
  list.EmplaceBack(CudaCopyD2HStreamType().Copy(dst_symbol, src_symbol, size));
  scheduler->Receive(&list);
  size_t count = 0;
  while (!scheduler->Empty()) {
    scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(scheduler->mut_vm_thread_list(), t) { t->TryReceiveAndRun(); }
    ++count;
  }
}

TEST(CudaCopyD2HStreamType, basic) { TestSimple(1); }

TEST(CudaCopyD2HStreamType, two_device) { TestSimple(2); }

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
